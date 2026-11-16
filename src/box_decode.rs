//! Bounding-box decoding, 4D Halton Monte-Carlo IoU confidence, and multi-box readout.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::calibration::Calibration;
use crate::config::BoxEdgeRead;
use crate::grid::{Grid, GRID_128, THOUSANDTHS_GRID};

/// Number of deterministic 4D Halton draws used to estimate `P(IoU >= 0.5)`.
pub const HALTON_DRAWS: usize = 512;

/// Default maximum instance blocks for a `boxes` question when `max` is omitted.
pub const DEFAULT_BOX_INSTANCES: usize = 16;

/// Hard upper bound on `boxes` instance blocks in prompt v2.
pub const MAX_BOX_INSTANCES: usize = 64;

/// Canonical edge slot order for `box` and `boxes` questions.
pub const EDGE_NAMES: [&str; 4] = ["top", "bottom", "left", "right"];

/// Normalized `[0.0, 1.0]` bounding box in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedBox {
    /// Distance from the top edge (`0.0..=1.0`).
    pub top: f64,
    /// Distance from the top edge (`top..=1.0`).
    pub bottom: f64,
    /// Distance from the left edge (`0.0..=1.0`).
    pub left: f64,
    /// Distance from the left edge (`left..=1.0`).
    pub right: f64,
}

impl NormalizedBox {
    /// Area of the normalized rectangle.
    #[must_use]
    pub fn area(&self) -> f64 {
        (self.bottom - self.top) * (self.right - self.left)
    }

    /// Intersection-over-Union (`IoU`) between two normalized rectangles.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f64 {
        let inter_h = (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0);
        let inter_w = (self.right.min(other.right) - self.left.max(other.left)).max(0.0);
        let overlap = inter_h * inter_w;
        let denom = self.area() + other.area() - overlap;
        if denom <= 0.0 {
            0.0
        } else {
            overlap / denom
        }
    }
}

/// Detected instance entry inside a `boxes` answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedBoxItem {
    /// Normalized rectangle coordinates.
    pub r#box: NormalizedBox,
    /// Calibrated probability that at least this many instances are present.
    pub found: f64,
    /// Calibrated probability that a sampled edge box overlaps with `IoU >= 0.5`.
    pub confidence: f64,
}

/// Ensures two coordinates along one axis are strictly ordered, expanding crossed edges to 1 bin.
#[must_use]
pub fn ordered_pair(first: f64, second: f64, grid: &Grid) -> (f64, f64) {
    if first < second {
        return (first, second);
    }
    let half = grid.step() / 2.0;
    let middle = ((first + second) / 2.0).clamp(half, 1.0 - half);
    (middle - half, middle + half)
}

/// Constructs a valid [`NormalizedBox`] from `[top, bottom, left, right]` edge coordinates.
#[must_use]
pub fn rectangle_from_edges(edges: [f64; 4], grid: &Grid) -> NormalizedBox {
    let (top, bottom) = ordered_pair(edges[0], edges[1], grid);
    let (left, right) = ordered_pair(edges[2], edges[3], grid);
    NormalizedBox {
        top,
        bottom,
        left,
        right,
    }
}

/// Selects the grid matching an edge distribution's length (`1000` thousandths vs `128` codes).
#[must_use]
pub fn grid_for_distribution(len: usize) -> Grid {
    if len == THOUSANDTHS_GRID.size() {
        THOUSANDTHS_GRID
    } else {
        GRID_128
    }
}

/// Interpolates where `probabilities`'s cumulative mass crosses `0.5` across uniform bins.
#[must_use]
pub fn cdf_crossing(probabilities: &[f64], grid: &Grid) -> f64 {
    let total: f64 = probabilities.iter().copied().sum();
    if total <= 0.0 {
        return grid.high();
    }
    let step = grid.step();
    let mut mass = 0.0;
    for (i, &p) in probabilities.iter().enumerate().take(grid.size()) {
        let share = p / total;
        if share > 0.0 && mass + share >= 0.5 {
            let val = grid.value_at(i);
            let interp = val - step / 2.0 + ((0.5 - mass) / share) * step;
            return interp.clamp(grid.low(), grid.high());
        }
        mass += share;
    }
    grid.high()
}

/// Collapses four edge distributions (`[top, bottom, left, right]`) into a [`NormalizedBox`].
#[must_use]
pub fn median_box(edges: &[Vec<f64>], mode: BoxEdgeRead) -> NormalizedBox {
    let grid = grid_for_distribution(edges[0].len());
    let mut coords = [0.0_f64; 4];
    for (idx, row) in edges.iter().enumerate().take(4) {
        coords[idx] = match mode {
            BoxEdgeRead::Interp => cdf_crossing(row, &grid),
            BoxEdgeRead::Median => grid.read(row, None::<fn(f64) -> f64>).median,
        };
    }
    rectangle_from_edges(coords, &grid)
}

fn radical_inverse(mut index: usize, base: usize) -> f64 {
    let mut value = 0.0;
    let mut scale = 1.0;
    let base_f = base as f64;
    while index > 0 {
        let digit = index % base;
        index /= base;
        scale /= base_f;
        value += (digit as f64) * scale;
    }
    value
}

fn halton_samples() -> &'static [[f64; 4]; HALTON_DRAWS] {
    static SAMPLES: OnceLock<[[f64; 4]; HALTON_DRAWS]> = OnceLock::new();
    SAMPLES.get_or_init(|| {
        let mut arr = [[0.0_f64; 4]; HALTON_DRAWS];
        for (idx, slot) in arr.iter_mut().enumerate() {
            let i = idx + 1;
            *slot = [
                radical_inverse(i, 2),
                radical_inverse(i, 3),
                radical_inverse(i, 5),
                radical_inverse(i, 7),
            ];
        }
        arr
    })
}

/// Computes the share of 4D Halton draws from `edges` achieving `IoU >= 0.5` with `target`.
#[must_use]
pub fn box_confidence(edges: &[Vec<f64>], target: &NormalizedBox) -> f64 {
    let grid = grid_for_distribution(edges[0].len());
    let values = grid.values();

    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .take(4)
        .map(|row| {
            let total: f64 = row.iter().copied().sum();
            let norm = if total > 0.0 { total } else { 1.0 };
            let mut acc = 0.0;
            row.iter()
                .map(|&p| {
                    acc += p / norm;
                    acc
                })
                .collect()
        })
        .collect();

    let mut hits = 0usize;
    for point in halton_samples() {
        let mut drawn = [0.0_f64; 4];
        for d in 0..4 {
            let cdf = &cdfs[d];
            let u = point[d];
            let pos = cdf.partition_point(|&c| c < u).min(grid.size() - 1);
            drawn[d] = values[pos];
        }
        if rectangle_from_edges(drawn, &grid).iou(target) >= 0.5 {
            hits += 1;
        }
    }

    (hits as f64) / (HALTON_DRAWS as f64)
}

/// Formats a single `box` slot group (`[presence, top, bottom, left, right]`) into its calibrated answer.
#[must_use]
pub fn decode_single_box(
    slot_probs: &[Vec<f64>],
    calibration: Option<&Calibration>,
    edge_mode: BoxEdgeRead,
) -> (Option<NormalizedBox>, f64, f64) {
    let raw_found = slot_probs
        .first()
        .and_then(|p| p.first().copied())
        .unwrap_or(0.0);
    let found = match calibration {
        Some(cal) => cal.through("found:box", raw_found),
        None => raw_found,
    };
    if found < 0.5 || slot_probs.len() < 5 {
        return (None, found, 0.0);
    }
    let edges = &slot_probs[1..5];
    let rect = median_box(edges, edge_mode);
    let raw_conf = box_confidence(edges, &rect);
    let confidence = match calibration {
        Some(cal) => cal.through("box", raw_conf),
        None => raw_conf,
    };
    (Some(rect), found, confidence)
}

/// Formats a multi-instance `boxes` slot group (`5 * blocks` distributions) in reading order.
#[must_use]
pub fn decode_many_boxes(
    slot_probs: &[Vec<f64>],
    calibration: Option<&Calibration>,
    edge_mode: BoxEdgeRead,
) -> (Vec<DetectedBoxItem>, bool, Vec<f64>) {
    let mut presence = Vec::with_capacity(slot_probs.len() / 5);
    let mut running_min = 1.0_f64;

    for chunk in slot_probs.chunks_exact(5) {
        let raw_p = chunk[0].first().copied().unwrap_or(0.0);
        let cal_p = match calibration {
            Some(cal) => cal.through("found:boxes", raw_p),
            None => raw_p,
        };
        running_min = running_min.min(cal_p);
        presence.push(running_min);
    }

    let mut detected = Vec::new();
    for (k, &found) in presence.iter().enumerate() {
        if found < 0.5 {
            break;
        }
        let edges = &slot_probs[5 * k + 1..5 * k + 5];
        let rect = median_box(edges, edge_mode);
        let raw_conf = box_confidence(edges, &rect);
        let confidence = match calibration {
            Some(cal) => cal.through("boxes", raw_conf),
            None => raw_conf,
        };
        detected.push(DetectedBoxItem {
            r#box: rect,
            found,
            confidence,
        });
    }

    detected.sort_by(|a, b| {
        a.r#box
            .left
            .total_cmp(&b.r#box.left)
            .then_with(|| a.r#box.top.total_cmp(&b.r#box.top))
    });

    let at_least = !detected.is_empty() && detected.len() == presence.len();
    (detected, at_least, presence)
}
