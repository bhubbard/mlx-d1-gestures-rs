//! Error types for the `mlx-d1-gestures` System-One engine and HTTP server.

use std::path::PathBuf;
use thiserror::Error;

/// Convenience result alias returning [`D1Error`].
pub type Result<T> = std::result::Result<T, D1Error>;

/// Errors produced while loading checkpoints, preprocessing frames, or running System-One passes.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum D1Error {
    /// Filesystem I/O failure with path context.
    #[error("failed to read or write {path}: {source}")]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// JSON configuration or schema serialization failure.
    #[error("invalid JSON configuration: {0}")]
    Json(#[from] serde_json::Error),

    /// Checkpoint configuration or question validation failure.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Caller question or frame validation failure.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Image decoding or resizing failure.
    #[error("invalid image frame: {0}")]
    Image(#[from] image::ImageError),

    /// HuggingFace tokenizer failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Missing tensor key in checkpoint SafeTensors shards.
    #[error("missing checkpoint weight tensor `{0}`")]
    MissingWeight(String),

    /// MLX tensor or Metal kernel execution failure.
    #[error("mlx evaluation error: {0}")]
    Mlx(String),

    /// HuggingFace Hub download or cache resolution failure.
    #[error("hub download failed: {0}")]
    Hub(String),
}

impl From<mlx_rs::error::Exception> for D1Error {
    #[doc(hidden)]
    fn from(err: mlx_rs::error::Exception) -> Self {
        Self::Mlx(err.to_string())
    }
}

impl From<mlx_rs::error::IoError> for D1Error {
    #[doc(hidden)]
    fn from(err: mlx_rs::error::IoError) -> Self {
        Self::Mlx(err.to_string())
    }
}

impl From<mlx_rs::ops::QuantizationError> for D1Error {
    #[doc(hidden)]
    fn from(err: mlx_rs::ops::QuantizationError) -> Self {
        Self::Mlx(err.to_string())
    }
}
