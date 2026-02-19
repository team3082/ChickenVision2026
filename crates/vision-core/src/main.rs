mod camera;
mod config;
mod detection;
mod streaming;

use config::Config;
use ndarray::Array2;
use std::time::Duration;
use tokio::time::Instant;
use tokio::runtime::Runtime;
use vision_detection::circle::precompute_circle_points;
use serde_json;
use std::net::{Ipv4Addr, SocketAddrV4};
pub use network_tables::v4::subscription::SubscriptionOptions;
//pub use network_tables::*;
use crate::{
    camera::{capture_frame, get_camera, resize_array},
    detection::{detect_circles, detect_contours, run_color_mask},
    streaming::{array_to_jpeg, run_dashboard_server, FrameHub},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_thread_ids(true)
        .with_env_filter("debug,network_tables=debug")
        .with_thread_names(true)
        .init();
    tracing::info!("RustyVision waking up...");

    // Load config
    let config = Config::load_default().unwrap_or_else(|e| {
        tracing::warn!(error = %e, "Using default configuration");
        Config::default()
    });
    // Initialize networktables
    let client = network_tables::v4::Client::try_new_w_config(
        //SocketAddrV4::new(Ipv4Addr::new(10, 30, 82, 2), 5810),
        SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 5810),
        network_tables::v4::client_config::Config {
            ..Default::default()
        },
    ).await?;
    let published_topic = client
        .publish_topic("/YellowOrbs", network_tables::v4::Type::Json, None)
        .await?;
    // Create FrameHubs for streaming

    let mask_hub = FrameHub::new();
    let contour_hub = FrameHub::new();
    let circle_hub = FrameHub::new();

    let state = run_dashboard_server(
        mask_hub.clone(),
        contour_hub.clone(),
        circle_hub.clone(),
        config.clone(),
    )
    .await?;
    let vision_state = state.clone();

    // Read constants from config
    let resize_factor: usize = 2;
    let width = config.camera.width as usize;
    let height = config.camera.height as usize;
    let proc_width = width / resize_factor;
    let proc_height = height / resize_factor;
    let rt  = Runtime::new().expect("Failed to create Tokio runtime");
    // Run vision processing in blocking task
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let mut camera = get_camera(vision_state.config.blocking_read().camera.device_id)?;
        camera.open_stream()?;

        let mut current_detection = vision_state.config.blocking_read().detection.clone();

        let mut circle_cache = precompute_circle_points(
            current_detection.min_radius / resize_factor as u32,
            current_detection.max_radius / resize_factor as u32,
            current_detection.radius_step,
        );

        // Buffers
        let mut rgb_frame: Array2<[u8; 3]> = Array2::from_elem((height, width), [0u8; 3]);
        let mut rgb_resized: Array2<[u8; 3]> =
            Array2::from_elem((proc_height, proc_width), [0u8; 3]);
        let mut mask_arr: Array2<u8> = Array2::zeros((proc_height, proc_width));
        let mut contour_arr: Array2<u8> = Array2::zeros((proc_height, proc_width));
        let mut circle_arr: Array2<u8> = Array2::zeros((proc_height, proc_width));

        let mut frame_counter = 0u32;
        let mut last_log = Instant::now();
        let task_client = client.clone();
        loop {
            // --- CONFIG UPDATE CHECK ---
            {
                let latest_config = vision_state.config.blocking_read();
                let latest_det = &latest_config.detection;

                if latest_det != &current_detection {
                    tracing::info!("Config update detected, applying new settings...");

                    // Radii change requires recomputing circles (expensive)
                    if latest_det.min_radius != current_detection.min_radius
                        || latest_det.max_radius != current_detection.max_radius
                        || latest_det.radius_step != current_detection.radius_step
                    {
                        circle_cache = precompute_circle_points(
                            latest_det.min_radius / resize_factor as u32,
                            latest_det.max_radius / resize_factor as u32,
                            latest_det.radius_step,
                        );
                    }

                    current_detection = latest_det.clone();
                }
            }

            // --- VISION PIPELINE ---
            // Camera capture into RGB buf
            capture_frame(&mut camera, &mut rgb_frame)?;
            resize_array(rgb_frame.view(), &mut rgb_resized, proc_height, proc_width);

            run_color_mask(
                rgb_resized.view(),
                &mut mask_arr,
                current_detection.color_lower,
                current_detection.color_upper,
            );

            detect_contours(
                mask_arr.view(),
                &mut contour_arr,
                current_detection.min_contour_length / resize_factor as u32,
                current_detection.min_area / (resize_factor * resize_factor) as f32,
            );

            let positions: Vec<(f64,f64,f64)> = detect_circles(
                contour_arr.view(),
                &mut circle_arr,
                &circle_cache,
                current_detection.vote_thresh,
            );

            // --- Publish to networktables
            let publishError = rt.block_on(publish(positions, task_client.clone(), published_topic.clone()));
            
            // --- PUBLISH TO DASHBOARD ---

            if vision_state.mask_frames.has_subscribers() {
                if let Some(jpeg) = array_to_jpeg(mask_arr.view()) {
                    vision_state.mask_frames.publish(jpeg);
                }
            }
            if vision_state.contour_frames.has_subscribers() {
                if let Some(jpeg) = array_to_jpeg(contour_arr.view()) {
                    vision_state.contour_frames.publish(jpeg);
                }
            }
            if vision_state.circle_frames.has_subscribers() {
                if let Some(jpeg) = array_to_jpeg(circle_arr.view()) {
                    vision_state.circle_frames.publish(jpeg);
                }
            }

            // --- FPS Logging ---
            frame_counter += 1;
            if last_log.elapsed() >= Duration::from_secs(1) {
                let fps = frame_counter as f64 / last_log.elapsed().as_secs_f64();
                tracing::info!("Stream FPS: {:.1}", fps);
                frame_counter = 0;
                last_log = Instant::now();
            }
        }
    })
    .await??;
    Ok(())
}

async fn publish( positions: Vec<(f64,f64,f64)>, task_client: network_tables::v4::Client, published_topic: network_tables::v4::PublishedTopic) -> anyhow::Result<()> {
    let json_data = serde_json::to_string(&positions)?;
    task_client.publish_value(&published_topic, &network_tables::Value::from(json_data))
        .await
        .unwrap();
    Ok(())
}