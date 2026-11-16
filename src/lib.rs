//! Pure-Rust Apple Silicon MLX port of `mlx-d1-gestures` (`LiquidAI/d1-2B` System-One webcam gestures).
//!
//! Runs `LiquidAI/d1-2B-MLX-8bit` (LFM3-VL vision encoder + multimodal projector + 24-layer LFM3
//! hybrid ShortConv/GQA language backbone under `set` attention) natively via `mlx-rs`, answering
//! hand gesture, facial expression, and normalized bounding-box questions in a single zero-token
//! forward pass on `127.0.0.1`.

pub mod box_decode;
pub mod calibration;
pub mod config;
pub mod engine;
pub mod error;
pub mod grid;
pub mod hub;
pub mod image_proc;
pub mod language;
pub mod mlx_layers;
pub mod prompt;
pub mod questions;
pub mod server;
pub mod vision;

pub use box_decode::{DetectedBoxItem, NormalizedBox};
pub use calibration::{Calibration, CalibrationKind};
pub use config::{BoxEdgeRead, ImageProcessorConfig, ModelConfig, ProcessorConfig};
pub use engine::{D1Engine, DecisionResponse, DecisionUsage};
pub use error::{D1Error, Result};
pub use grid::{Grid, GridStats, GRID_128, THOUSANDTHS_GRID};
pub use questions::{default_gesture_questions, Answer, BinaryCriteria, Question, ScoreReadout};
pub use server::{
    build_router, serve_loopback, ServerHandle, StatusResponse, DEFAULT_DECIDE_PORT,
    MAX_FRAME_BYTES,
};
