//! Loopback HTTP server (`127.0.0.1:8049`) with a dedicated single-flight Metal GPU worker thread.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, Request, StatusCode};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, RwLock};

use crate::engine::{D1Engine, DecisionResponse};
use crate::error::{D1Error, Result};
use crate::questions::Question;

/// Default loopback TCP port matching upstream `decide.py` (`DECIDE_PORT=8049`).
pub const DEFAULT_DECIDE_PORT: u16 = 8049;

/// Maximum raw frame upload body (`4 MiB`), matching upstream `MAX_FRAME = 4 * 1024 * 1024`.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

const EMBEDDED_INDEX_HTML: &str = include_str!("../assets/index.html");

/// JSON payload returned by `GET /status`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Active checkpoint identifier.
    pub model: String,
    /// `true` once model weights are loaded and warmed up on Metal.
    pub ready: bool,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    error: String,
}

struct FrameJob {
    frame_bytes: Vec<u8>,
    questions: IndexMap<String, Question>,
    reply: oneshot::Sender<Result<DecisionResponse>>,
}

/// Thread-safe handle routing incoming HTTP frames sequentially to the dedicated MLX worker thread.
#[derive(Clone)]
pub struct ServerHandle {
    model_id: Arc<str>,
    questions: Arc<RwLock<IndexMap<String, Question>>>,
    tx: mpsc::Sender<FrameJob>,
}

impl std::fmt::Debug for ServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerHandle")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ServerHandle>();
};

impl ServerHandle {
    /// Spawns a dedicated OS worker thread that builds and warms up [`D1Engine`] in-place,
    /// keeping MLX `!Send` array handles strictly confined to one thread with zero `unsafe`.
    ///
    /// Returns both the [`ServerHandle`] and the warm-up [`DecisionResponse`].
    ///
    /// # Errors
    /// Returns [`D1Error`] if the factory closure or warm-up pass fails.
    pub fn try_spawn<F>(
        initial_questions: IndexMap<String, Question>,
        factory: F,
    ) -> Result<(Self, DecisionResponse)>
    where
        F: FnOnce() -> Result<D1Engine> + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel::<FrameJob>(4);
        let (boot_tx, boot_rx) =
            std::sync::mpsc::sync_channel::<Result<(String, DecisionResponse)>>(1);
        let warm_questions = initial_questions.clone();

        std::thread::Builder::new()
            .name("d1-mlx-gpu-worker".to_string())
            .spawn(move || {
                let mut engine = match factory() {
                    Ok(eng) => eng,
                    Err(err) => {
                        let _ = boot_tx.send(Err(err));
                        return;
                    }
                };
                match engine.warmup(&warm_questions) {
                    Ok(warm) => {
                        let _ = boot_tx.send(Ok((engine.model_id().to_string(), warm)));
                    }
                    Err(err) => {
                        let _ = boot_tx.send(Err(err));
                        return;
                    }
                }
                while let Some(job) = rx.blocking_recv() {
                    let out = engine.decide_encoded_frame(&job.frame_bytes, &job.questions);
                    let _ = job.reply.send(out);
                }
            })
            .map_err(|source| D1Error::Io {
                path: std::path::PathBuf::from("d1-mlx-gpu-worker"),
                source,
            })?;

        let (model_id, warmup_resp) = boot_rx
            .recv()
            .map_err(|_| D1Error::Mlx("GPU worker exited during initialization".to_string()))??;

        Ok((
            Self {
                model_id: Arc::from(model_id),
                questions: Arc::new(RwLock::new(initial_questions)),
                tx,
            },
            warmup_resp,
        ))
    }

    /// Evaluates one encoded JPEG/PNG frame against the active question map.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the worker thread terminated or frame inference fails.
    pub async fn decide_frame(&self, frame_bytes: Vec<u8>) -> Result<DecisionResponse> {
        let questions = self.questions.read().await.clone();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(FrameJob {
                frame_bytes,
                questions,
                reply: reply_tx,
            })
            .await
            .map_err(|_| D1Error::Mlx("GPU worker channel closed".to_string()))?;

        reply_rx
            .await
            .map_err(|_| D1Error::Mlx("GPU worker dropped response".to_string()))?
    }
}

/// Builds the Axum [`Router`] implementing `/`, `/index.html`, `/status`, `/questions`, and `/run`.
pub fn build_router(handle: ServerHandle) -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/index.html", get(serve_index))
        .route("/status", get(serve_status))
        .route("/questions", get(get_questions).post(update_questions))
        .route("/run", post(handle_run))
        .fallback(handle_not_found)
        // Allow slightly over 4 MiB into `handle_run` so we return exact 413 JSON like `decide.py`.
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES + 1024 * 1024))
        .with_state(handle)
}

/// Serves the webcam UI and `POST /run` endpoint strictly on `127.0.0.1:port` until Ctrl-C.
///
/// # Errors
/// Returns [`D1Error`] if binding the loopback TCP listener fails.
pub async fn serve_loopback(handle: ServerHandle, port: u16) -> Result<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| D1Error::Io {
            path: std::path::PathBuf::from(addr.to_string()),
            source,
        })?;
    let app = build_router(handle);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|source| D1Error::Io {
            path: std::path::PathBuf::from(addr.to_string()),
            source,
        })
}

async fn serve_index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(EMBEDDED_INDEX_HTML),
    )
}

async fn serve_status(State(handle): State<ServerHandle>) -> Json<StatusResponse> {
    Json(StatusResponse {
        model: handle.model_id.to_string(),
        ready: true,
    })
}

async fn get_questions(State(handle): State<ServerHandle>) -> Json<IndexMap<String, Question>> {
    let qs = handle.questions.read().await.clone();
    Json(qs)
}

async fn update_questions(
    State(handle): State<ServerHandle>,
    Json(next): Json<IndexMap<String, Question>>,
) -> Response {
    if next.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "questions map cannot be empty");
    }
    *handle.questions.write().await = next.clone();
    (StatusCode::OK, Json(next)).into_response()
}

async fn handle_run(State(handle): State<ServerHandle>, request: Request<Body>) -> Response {
    if let Some(cl) = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if cl == 0 || cl > MAX_FRAME_BYTES {
            return json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "send one JPEG frame, up to 4 MB",
            );
        }
    }

    let bytes = match to_bytes(request.into_body(), MAX_FRAME_BYTES + 1).await {
        Ok(b) => b,
        Err(_) => {
            return json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "send one JPEG frame, up to 4 MB",
            );
        }
    };

    if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
        return json_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "send one JPEG frame, up to 4 MB",
        );
    }

    match handle.decide_frame(bytes.to_vec()).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(err) => json_error(StatusCode::BAD_REQUEST, &err.to_string()),
    }
}

async fn handle_not_found() -> Response {
    json_error(StatusCode::NOT_FOUND, "unknown path")
}

fn json_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorPayload {
            error: message.to_string(),
        }),
    )
        .into_response()
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
