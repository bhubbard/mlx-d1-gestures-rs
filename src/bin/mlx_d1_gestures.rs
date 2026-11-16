//! CLI entrypoint for `mlx-d1-gestures`: local loopback webcam server and one-shot frame classifier.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indexmap::IndexMap;

use mlx_d1_gestures::{
    default_gesture_questions, serve_loopback, D1Engine, Question, ScoreReadout, ServerHandle,
    DEFAULT_DECIDE_PORT,
};

const DEFAULT_MODEL_ID: &str = "LiquidAI/d1-2B-MLX-8bit";

#[derive(Debug, Parser)]
#[command(
    name = "mlx-d1-gestures",
    version,
    about = "Live webcam hand gesture & facial expression recognition in one zero-token pass on Apple Silicon MLX"
)]
struct Cli {
    /// HuggingFace repo ID or local checkpoint directory (`DECIDE_MODEL`).
    #[arg(long, env = "DECIDE_MODEL", default_value = DEFAULT_MODEL_ID, global = true)]
    model: String,

    /// Optional path to a custom JSON map of named System-One questions.
    #[arg(long, global = true)]
    questions: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,

    /// Loopback HTTP port (`DECIDE_PORT`) when running in default server mode.
    #[arg(long, env = "DECIDE_PORT", default_value_t = DEFAULT_DECIDE_PORT)]
    port: u16,

    /// Open `http://127.0.0.1:<port>` in the default macOS browser once ready (replaces `start.sh`).
    #[arg(long)]
    open: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start the `127.0.0.1` live webcam gesture server (default when no subcommand is given).
    Serve {
        /// Loopback HTTP port (`DECIDE_PORT`).
        #[arg(long, env = "DECIDE_PORT", default_value_t = DEFAULT_DECIDE_PORT)]
        port: u16,
        /// Open the webcam UI in the default browser once the warm-up pass completes.
        #[arg(long)]
        open: bool,
    },
    /// Run a single zero-token forward pass on a still image file and print JSON to stdout.
    Decide {
        /// Path to an input JPEG, PNG, or WebP frame.
        image: PathBuf,
        /// Optional text context accompanying the image.
        #[arg(long)]
        state: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let questions = load_questions(cli.questions.as_deref())?;

    match cli.command {
        Some(Commands::Decide { image, state }) => {
            eprintln!("loading {}...", cli.model);
            let mut engine = D1Engine::load(&cli.model)
                .with_context(|| format!("loading checkpoint {}", cli.model))?;
            let img = image::open(&image)
                .with_context(|| format!("opening image {}", image.display()))?;
            let response = engine
                .decide_image(&img, state.as_deref(), &questions, ScoreReadout::Median)
                .context("running System-One decision pass")?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Some(Commands::Serve { port, open }) => {
            run_server(&cli.model, questions, port, open).await?;
        }
        None => {
            run_server(&cli.model, questions, cli.port, cli.open).await?;
        }
    }

    Ok(())
}

async fn run_server(
    model_id: &str,
    questions: IndexMap<String, Question>,
    port: u16,
    open_browser: bool,
) -> Result<()> {
    println!(
        "loading {model_id} (the first run downloads ~2.2 GB into ~/.cache/huggingface/hub)..."
    );
    let owned_model_id = model_id.to_string();
    let (handle, warm) =
        ServerHandle::try_spawn(questions, move || D1Engine::load(&owned_model_id))
            .with_context(|| format!("loading and warming up checkpoint {model_id}"))?;

    println!(
        "warm-up complete ({:.1} ms, {} prefill tokens, 0 output tokens)",
        warm.ms, warm.usage.input_tokens
    );

    let url = format!("http://127.0.0.1:{port}");
    println!("d1 ready: open {url}");

    if open_browser && cfg!(target_os = "macos") {
        let _ = Command::new("open").arg(&url).spawn();
    }

    serve_loopback(handle, port)
        .await
        .context("running loopback HTTP server")
}

fn load_questions(path: Option<&Path>) -> Result<IndexMap<String, Question>> {
    let Some(p) = path else {
        return Ok(default_gesture_questions());
    };
    let raw =
        fs::read_to_string(p).with_context(|| format!("reading questions file {}", p.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing questions JSON {}", p.display()))
}
