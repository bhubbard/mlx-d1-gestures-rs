//! Pure-Rust HuggingFace Hub cache resolution and streaming checkpoint downloader.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

use crate::error::{D1Error, Result};

const REQUIRED_META_FILES: [&str; 4] = [
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

const OPTIONAL_META_FILES: [&str; 2] = ["calibration.json", "model.safetensors.index.json"];

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Resolves `model_spec` to a local checkpoint directory, downloading from HuggingFace Hub if needed.
///
/// Checks existing local paths and `~/.cache/huggingface/hub` snapshots first so offline cached
/// runs require zero network calls.
///
/// # Errors
/// Returns [`D1Error`] if `model_spec` cannot be found locally or downloaded from HuggingFace Hub.
pub fn resolve_checkpoint_dir(model_spec: &str) -> Result<PathBuf> {
    let direct = Path::new(model_spec);
    if direct.is_dir() && direct.join("config.json").is_file() {
        return Ok(direct.to_path_buf());
    }

    let cache_root = hf_hub_cache_root();
    let repo_folder = format!("models--{}", model_spec.replace('/', "--"));
    let repo_cache = cache_root.join(repo_folder);

    if let Some(existing) = find_complete_snapshot(&repo_cache) {
        return Ok(existing);
    }

    let snapshot_dir = repo_cache.join("snapshots").join("main");
    fs::create_dir_all(&snapshot_dir).map_err(|source| D1Error::Io {
        path: snapshot_dir.clone(),
        source,
    })?;

    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok();

    for file in REQUIRED_META_FILES {
        download_if_missing(model_spec, file, &snapshot_dir, token.as_deref(), true)?;
    }
    for file in OPTIONAL_META_FILES {
        download_if_missing(model_spec, file, &snapshot_dir, token.as_deref(), false)?;
    }

    let index_path = snapshot_dir.join("model.safetensors.index.json");
    let shard_files: BTreeSet<String> = if index_path.is_file() {
        let raw = fs::read_to_string(&index_path).map_err(|source| D1Error::Io {
            path: index_path,
            source,
        })?;
        let idx: SafetensorsIndex = serde_json::from_str(&raw)?;
        idx.weight_map.into_values().collect()
    } else {
        BTreeSet::from(["model.safetensors".to_string()])
    };

    for shard in shard_files {
        download_if_missing(model_spec, &shard, &snapshot_dir, token.as_deref(), true)?;
    }

    Ok(snapshot_dir)
}

fn hf_hub_cache_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(explicit);
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn find_complete_snapshot(repo_cache: &Path) -> Option<PathBuf> {
    let snapshots = repo_cache.join("snapshots");
    let entries = fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        if !dir.join("config.json").is_file() || !dir.join("tokenizer.json").is_file() {
            continue;
        }
        let has_safetensors = fs::read_dir(&dir)
            .ok()?
            .flatten()
            .any(|f| f.path().extension().and_then(|e| e.to_str()) == Some("safetensors"));
        if has_safetensors {
            return Some(dir);
        }
    }
    None
}

fn download_if_missing(
    repo_id: &str,
    filename: &str,
    dest_dir: &Path,
    token: Option<&str>,
    required: bool,
) -> Result<()> {
    let dest = dest_dir.join(filename);
    if dest.is_file() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|source| D1Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let base =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let url = format!(
        "{}/{}",
        base.trim_end_matches('/'),
        format_args!("{repo_id}/resolve/main/{filename}")
    );

    let mut req = ureq::get(&url);
    if let Some(tok) = token {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }

    let response = match req.call() {
        Ok(resp) => resp,
        Err(ureq::Error::StatusCode(404)) if !required => return Ok(()),
        Err(err) => {
            return Err(D1Error::Hub(format!("GET {url} failed: {err}")));
        }
    };

    let total_bytes = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let pb = if total_bytes > 1024 * 1024 {
        let bar = ProgressBar::new(total_bytes);
        if let Ok(style) = ProgressStyle::with_template(
            "{msg} [{bar:36.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})",
        ) {
            bar.set_style(style.progress_chars("=>-"));
        }
        bar.set_message(format!("downloading {filename}"));
        Some(bar)
    } else {
        None
    };

    let part_path = dest_dir.join(format!("{filename}.part"));
    let file = File::create(&part_path).map_err(|source| D1Error::Io {
        path: part_path.clone(),
        source,
    })?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);
    let mut reader = response.into_body().into_reader();
    let mut buf = vec![0_u8; 128 * 1024];

    loop {
        let n = reader.read(&mut buf).map_err(|source| D1Error::Io {
            path: part_path.clone(),
            source,
        })?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n]).map_err(|source| D1Error::Io {
            path: part_path.clone(),
            source,
        })?;
        if let Some(bar) = &pb {
            bar.inc(n as u64);
        }
    }
    writer.flush().map_err(|source| D1Error::Io {
        path: part_path.clone(),
        source,
    })?;
    drop(writer);

    if let Some(bar) = pb {
        bar.finish_and_clear();
    }

    fs::rename(&part_path, &dest).map_err(|source| D1Error::Io { path: dest, source })?;
    Ok(())
}
