//! Temperature scaling and monotone knot calibration tables (`calibration.json`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{D1Error, Result};

/// Kind of discrete option readout used to select a temperature bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationKind {
    /// Binary `yes` / `no` readout.
    Noul,
    /// Ordinal `0..k` score readout.
    Score(usize),
    /// Categorical choice readout with `k` options.
    Choice(usize),
}

impl CalibrationKind {
    /// Canonical bucket key in `calibration.json` `temperatures`.
    #[must_use]
    pub fn bucket_key(self) -> String {
        match self {
            Self::Noul => "noul".to_string(),
            Self::Score(k) => format!("score:{k}"),
            Self::Choice(k) if k <= 3 => "choice:small".to_string(),
            Self::Choice(k) if k <= 8 => "choice:mid".to_string(),
            Self::Choice(k) if k <= 26 => "choice:large".to_string(),
            Self::Choice(_) => "choice:inventory".to_string(),
        }
    }
}

/// Interpolates `value` through increasing `[raw, calibrated]` knots, clamping outside ends.
#[must_use]
pub fn interpolate_knots(value: f64, knots: &[[f64; 2]]) -> f64 {
    if knots.is_empty() {
        return value;
    }
    let idx = knots.partition_point(|pair| pair[0] < value);
    if idx == 0 {
        return knots[0][1];
    }
    if idx >= knots.len() {
        return knots[knots.len() - 1][1];
    }
    let [a, pa] = knots[idx - 1];
    let [b, pb] = knots[idx];
    if (b - a).abs() <= f64::EPSILON {
        return pb;
    }
    pa + (value - a) * (pb - pa) / (b - a)
}

/// Post-hoc calibration parameters shipped alongside `LiquidAI/d1-2B` checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Per-bucket temperature overrides.
    #[serde(default)]
    pub temperatures: HashMap<String, f64>,
    /// Global fallback temperature (`1.3374` on `d1-2B`).
    #[serde(default = "default_temperature")]
    pub default: f64,
    /// Validation sample counts recorded during calibration fitting.
    #[serde(default)]
    pub counts: HashMap<String, usize>,
    /// Monotone `[raw, calibrated]` knot tables (`found:box`, `box`, `multiple`, `estimate`, etc.).
    #[serde(default)]
    pub tables: HashMap<String, Vec<[f64; 2]>>,
}

const fn default_temperature() -> f64 {
    1.0
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            temperatures: HashMap::new(),
            default: 1.0,
            counts: HashMap::new(),
            tables: HashMap::new(),
        }
    }
}

impl Calibration {
    /// Loads `calibration.json` from disk.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path).map_err(|source| D1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Loads `calibration.json` if it exists, or returns identity calibration (`T = 1.0`).
    ///
    /// # Errors
    /// Returns [`D1Error`] if the file exists but cannot be parsed.
    pub fn load_optional(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = path.as_ref();
        if !path.is_file() {
            return Ok(None);
        }
        Self::load(path).map(Some)
    }

    /// Temperature assigned to a question bucket.
    #[must_use]
    pub fn temperature(&self, kind: CalibrationKind) -> f64 {
        self.temperatures
            .get(&kind.bucket_key())
            .copied()
            .unwrap_or(self.default)
    }

    /// Scales raw option logits by `1 / T` in place when `T != 1.0`.
    pub fn scale_scores(&self, kind: CalibrationKind, scores: &mut [f64]) {
        let temp = self.temperature(kind);
        if (temp - 1.0).abs() > f64::EPSILON && temp > 0.0 {
            for s in scores {
                *s /= temp;
            }
        }
    }

    /// Maps a raw probability through the named calibration table (`"found:box"`, `"box"`, etc.).
    #[must_use]
    pub fn through(&self, table_name: &str, raw_prob: f64) -> f64 {
        match self.tables.get(table_name) {
            Some(knots) => interpolate_knots(raw_prob, knots),
            None => raw_prob,
        }
    }

    /// Maps a target quantile share backwards through a monotone table (`"estimate"`).
    #[must_use]
    pub fn level(&self, table_name: &str, share: f64) -> f64 {
        match self.tables.get(table_name) {
            Some(knots) => {
                let inverted: Vec<[f64; 2]> = knots.iter().map(|&[x, y]| [y, x]).collect();
                interpolate_knots(share, &inverted)
            }
            None => share,
        }
    }
}
