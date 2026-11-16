//! Equally spaced numeric grids for `d1` estimate and bounding-box edge readouts.

use crate::error::{D1Error, Result};

/// Summary statistics read from a discrete distribution over a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridStats {
    /// Expected value across all grid bins.
    pub mean: f64,
    /// 10th percentile value (or calibrated quantile).
    pub p10: f64,
    /// 50th percentile median value (or calibrated quantile).
    pub median: f64,
    /// 90th percentile value (or calibrated quantile).
    pub p90: f64,
}

/// Ordered values over `size` equally spaced bins from `low` to `high`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    low: f64,
    high: f64,
    size: usize,
}

/// Default 128-bin code grid over `[0.0, 1.0]`.
pub const GRID_128: Grid = Grid {
    low: 0.0,
    high: 1.0,
    size: 128,
};

/// Default 1,000-bin thousandths grid over `[0.0, 0.999]` (`"0"`..`"999"`).
pub const THOUSANDTHS_GRID: Grid = Grid {
    low: 0.0,
    high: 0.999,
    size: 1000,
};

impl Grid {
    /// Creates a validated [`Grid`] with `size >= 2` and `low < high`.
    ///
    /// # Errors
    /// Returns [`D1Error::InvalidRequest`] if `size < 2` or `!(low < high)`.
    pub fn try_new(low: f64, high: f64, size: usize) -> Result<Self> {
        if size < 2 || !low.is_finite() || !high.is_finite() || low >= high {
            return Err(D1Error::InvalidRequest(format!(
                "grid requires size >= 2 and finite low < high, got low={low}, high={high}, size={size}"
            )));
        }
        Ok(Self { low, high, size })
    }

    /// Lower bound of the grid.
    #[must_use]
    pub const fn low(&self) -> f64 {
        self.low
    }

    /// Upper bound of the grid.
    #[must_use]
    pub const fn high(&self) -> f64 {
        self.high
    }

    /// Number of bins in the grid.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Spacing between adjacent bin centers.
    #[must_use]
    pub fn step(&self) -> f64 {
        (self.high - self.low) / ((self.size - 1) as f64)
    }

    /// Coordinate of bin `index` (`0..size`).
    #[must_use]
    pub fn value_at(&self, index: usize) -> f64 {
        if index + 1 >= self.size {
            self.high
        } else {
            self.low + (index as f64) * self.step()
        }
    }

    /// Materializes all bin coordinates in ascending order.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.value_at(i)).collect()
    }

    /// Computes mean, median, and central 80% interval (`p10`, `p90`) over `probabilities`.
    #[must_use]
    pub fn read<F>(&self, probabilities: &[f64], level_map: Option<F>) -> GridStats
    where
        F: Fn(f64) -> f64,
    {
        let total: f64 = probabilities.iter().copied().sum();
        let norm = if total > 0.0 { total } else { 1.0 };

        let mut mean = 0.0;
        for (i, &p) in probabilities.iter().enumerate().take(self.size) {
            mean += self.value_at(i) * (p / norm);
        }

        let map_q = |q: f64| match &level_map {
            Some(f) => f(q),
            None => q,
        };
        let q10 = map_q(0.1);
        let q50 = map_q(0.5);
        let q90 = map_q(0.9);

        let mut p10 = None;
        let mut median = None;
        let mut p90 = None;
        let mut mass = 0.0;

        for (i, &p) in probabilities.iter().enumerate().take(self.size) {
            mass += p / norm;
            let val = self.value_at(i);
            if p10.is_none() && mass >= q10 {
                p10 = Some(val);
            }
            if median.is_none() && mass >= q50 {
                median = Some(val);
            }
            if p90.is_none() && mass >= q90 {
                p90 = Some(val);
            }
        }

        GridStats {
            mean,
            p10: p10.unwrap_or(self.high),
            median: median.unwrap_or(self.high),
            p90: p90.unwrap_or(self.high),
        }
    }
}
