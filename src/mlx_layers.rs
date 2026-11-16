//! MLX weight sanitization and unified dense / affine-quantized linear & embedding primitives.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{DequantizeOptions, QuantizedMatmulOptions};
use mlx_rs::{fast, ops, Array};

use crate::config::QuantizationConfig;
use crate::error::{D1Error, Result};

///Sanitized checkpoint tensor store loaded onto Apple Silicon unified memory.
#[derive(Debug)]
pub struct WeightStore {
    tensors: HashMap<String, Array>,
    quantization: Option<QuantizationConfig>,
}

impl WeightStore {
    /// Loads and sanitizes all `.safetensors` shards in `checkpoint_dir`.
    ///
    /// Applies LFM3-VL key canonicalization (`model.language_model.` -> `language_model.model.`)
    /// and transposes PyTorch channel-first depthwise convolution weights to MLX channel-last layout.
    ///
    /// # Errors
    /// Returns [`D1Error`] if no SafeTensors files exist or loading/transposition fails.
    pub fn load_from_dir(
        checkpoint_dir: impl AsRef<Path>,
        quantization: Option<QuantizationConfig>,
        tie_word_embeddings: bool,
    ) -> Result<Self> {
        let dir = checkpoint_dir.as_ref();
        let mut shard_paths = Vec::new();
        for entry in fs::read_dir(dir).map_err(|source| D1Error::Io {
            path: dir.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| D1Error::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                shard_paths.push(path);
            }
        }
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(D1Error::InvalidConfig(format!(
                "no .safetensors files found in {}",
                dir.display()
            )));
        }

        let mut raw_map = HashMap::new();
        for shard in shard_paths {
            let loaded = Array::load_safetensors(&shard)?;
            raw_map.extend(loaded);
        }

        Self::from_raw_tensors(raw_map, quantization, tie_word_embeddings)
    }

    /// Constructs a [`WeightStore`] from an in-memory map of tensors, applying LFM3-VL sanitization.
    ///
    /// # Errors
    /// Returns [`D1Error`] if tensor transposition fails.
    pub fn from_raw_tensors(
        raw_map: HashMap<String, Array>,
        quantization: Option<QuantizationConfig>,
        tie_word_embeddings: bool,
    ) -> Result<Self> {
        let mut tensors = HashMap::with_capacity(raw_map.len());
        for (raw_name, mut value) in raw_map {
            let name = if let Some(rest) = raw_name.strip_prefix("model.language_model.") {
                format!("language_model.model.{rest}")
            } else if let Some(rest) = raw_name.strip_prefix("model.") {
                rest.to_string()
            } else if raw_name.starts_with("lm_head.") {
                format!("language_model.{raw_name}")
            } else {
                raw_name
            };

            if tie_word_embeddings && name.starts_with("language_model.lm_head.") {
                continue;
            }

            if name.ends_with("conv.conv.weight")
                && value.ndim() == 3
                && value.shape()[1] == 1
                && value.shape()[2] != 1
            {
                value = value.transpose_axes(&[0, 2, 1])?;
            } else if name.ends_with("spatial_conv.weight")
                && value.ndim() == 4
                && value.shape()[1] == 1
                && value.shape()[3] != 1
            {
                value = value.transpose_axes(&[0, 2, 3, 1])?;
            }

            tensors.insert(name, value);
        }

        Ok(Self {
            tensors,
            quantization,
        })
    }

    /// Removes and returns a required tensor by exact name.
    ///
    /// # Errors
    /// Returns [`D1Error::MissingWeight`] if `key` is absent.
    pub fn take(&mut self, key: &str) -> Result<Array> {
        self.tensors
            .remove(key)
            .ok_or_else(|| D1Error::MissingWeight(key.to_string()))
    }

    /// Removes and returns an optional tensor if present.
    pub fn take_optional(&mut self, key: &str) -> Option<Array> {
        self.tensors.remove(key)
    }

    /// Loads either a quantized (`weight` + `scales` + `biases`) or dense [`LinearLayer`] at `prefix`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if `prefix.weight` is missing or quantization parameters are incomplete.
    pub fn take_linear(&mut self, prefix: &str) -> Result<LinearLayer> {
        let weight = self.take(&format!("{prefix}.weight"))?;
        let bias = self.take_optional(&format!("{prefix}.bias"));

        if let Some(scales) = self.take_optional(&format!("{prefix}.scales")) {
            let qcfg = self.quantization.unwrap_or(QuantizationConfig {
                group_size: 64,
                bits: 8,
            });
            let biases = self.take(&format!("{prefix}.biases"))?;
            Ok(LinearLayer::Quantized {
                weight,
                scales,
                biases,
                bias,
                group_size: qcfg.group_size,
                bits: qcfg.bits,
            })
        } else {
            Ok(LinearLayer::Dense { weight, bias })
        }
    }

    /// Loads either a quantized or dense [`EmbeddingLayer`] at `prefix`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if `prefix.weight` is missing.
    pub fn take_embedding(&mut self, prefix: &str) -> Result<EmbeddingLayer> {
        let weight = self.take(&format!("{prefix}.weight"))?;
        if let Some(scales) = self.take_optional(&format!("{prefix}.scales")) {
            let qcfg = self.quantization.unwrap_or(QuantizationConfig {
                group_size: 64,
                bits: 8,
            });
            let biases = self.take(&format!("{prefix}.biases"))?;
            Ok(EmbeddingLayer::Quantized {
                weight,
                scales,
                biases,
                group_size: qcfg.group_size,
                bits: qcfg.bits,
            })
        } else {
            Ok(EmbeddingLayer::Dense { weight })
        }
    }

    /// Loads an [`RmsNorm`] weight at `prefix`.
    ///
    /// # Errors
    /// Returns [`D1Error::MissingWeight`] if `prefix.weight` is missing.
    pub fn take_rms_norm(&mut self, prefix: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.take(&format!("{prefix}.weight"))?,
            eps,
        })
    }

    /// Loads a [`LayerNorm`] weight and bias at `prefix`.
    ///
    /// # Errors
    /// Returns [`D1Error::MissingWeight`] if `prefix.weight` or `prefix.bias` is missing.
    pub fn take_layer_norm(&mut self, prefix: &str, eps: f32) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: self.take(&format!("{prefix}.weight"))?,
            bias: self.take(&format!("{prefix}.bias"))?,
            eps,
        })
    }
}

/// Linear projection layer supporting both fp16/bf16 dense weights and 4/8-bit affine MLX quantization.
#[derive(Debug, Clone)]
pub enum LinearLayer {
    /// Unquantized `[out_features, in_features]` weight matrix.
    Dense {
        /// Weight matrix `[out, in]`.
        weight: Array,
        /// Optional additive bias `[out]`.
        bias: Option<Array>,
    },
    /// MLX affine-quantized weight matrix with per-group scales and biases.
    Quantized {
        /// Packed uint32 weight matrix.
        weight: Array,
        /// Per-group scale factors.
        scales: Array,
        /// Per-group additive offsets.
        biases: Array,
        /// Optional post-matmul bias `[out]`.
        bias: Option<Array>,
        /// Quantization group size (`64`).
        group_size: i32,
        /// Bit-width per weight (`8` or `4`).
        bits: i32,
    },
}

impl LinearLayer {
    /// Applies `x @ W^T (+ b)` on Metal unified memory.
    ///
    /// # Errors
    /// Returns [`D1Error`] on shape mismatch or MLX kernel failure.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense { weight, bias } => {
                let wt = weight.t();
                match bias {
                    Some(b) => Ok(ops::addmm(b, x, wt, None, None)?),
                    None => Ok(ops::matmul(x, wt)?),
                }
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            } => {
                let out = ops::quantized_matmul(
                    x,
                    weight,
                    scales,
                    Some(biases),
                    QuantizedMatmulOptions {
                        transpose: true,
                        group_size: Some(*group_size),
                        bits: Some(*bits),
                        ..Default::default()
                    },
                )?;
                match bias {
                    Some(b) => Ok(ops::add(&out, b)?),
                    None => Ok(out),
                }
            }
        }
    }
}

/// Token embedding table supporting dense lookup, quantized dequantize-on-gather, and candidate projection.
#[derive(Debug, Clone)]
pub enum EmbeddingLayer {
    /// Unquantized `[vocab_size, hidden_size]` embedding matrix.
    Dense {
        /// Embedding matrix `[V, D]`.
        weight: Array,
    },
    /// Affine-quantized `[vocab_size, hidden_size * bits / 32]` embedding matrix.
    Quantized {
        /// Packed uint32 weight matrix.
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group biases.
        biases: Array,
        /// Quantization group size (`64`).
        group_size: i32,
        /// Bit-width (`8` or `4`).
        bits: i32,
    },
}

impl EmbeddingLayer {
    /// Looks up embeddings for `token_ids` (`[T]` -> `[T, D]`).
    ///
    /// # Errors
    /// Returns [`D1Error`] if indexing or dequantization fails.
    pub fn embed(&self, token_ids: &Array) -> Result<Array> {
        match self {
            Self::Dense { weight } => Ok(weight.index(token_ids)),
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let flat = token_ids.flatten(None, None)?;
                let w = weight.index(&flat);
                let s = scales.index(&flat);
                let b = biases.index(&flat);
                let deq = ops::dequantize(
                    &w,
                    &s,
                    Some(&b),
                    DequantizeOptions {
                        group_size: Some(*group_size),
                        bits: Some(*bits),
                        ..Default::default()
                    },
                )?;
                let mut shape: Vec<i32> = token_ids.shape().to_vec();
                shape.push(-1);
                Ok(deq.reshape(&shape)?)
            }
        }
    }

    /// Projects slot hidden states `[S, D]` strictly against `candidate_ids` `[K]` -> `[S, K]`.
    ///
    /// Avoids computing the full `128,000`-token vocabulary projection on every decision pass.
    ///
    /// # Errors
    /// Returns [`D1Error`] if gathering or matrix multiplication fails.
    pub fn project_candidates(&self, hidden_slots: &Array, candidate_ids: &Array) -> Result<Array> {
        let candidate_rows = self.embed(candidate_ids)?.as_dtype(hidden_slots.dtype())?;
        Ok(ops::matmul(hidden_slots, candidate_rows.t())?)
    }
}

/// Fused Metal RMSNorm wrapper (`mlx_rs::fast::rms_norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Multiplicative scale vector `[D]`.
    pub weight: Array,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl RmsNorm {
    /// Normalizes `x` along its last dimension.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the Metal RMSNorm kernel fails.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::rms_norm(x, Some(&self.weight), self.eps)?)
    }
}

/// Fused Metal LayerNorm wrapper (`mlx_rs::fast::layer_norm`).
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Multiplicative weight `[D]`.
    pub weight: Array,
    /// Additive bias `[D]`.
    pub bias: Array,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl LayerNorm {
    /// Applies affine LayerNorm along the last dimension of `x`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the Metal LayerNorm kernel fails.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(x, &self.weight, &self.bias, self.eps)?)
    }
}

/// SwiGLU feed-forward block (`w2(silu(w1(x)) * w3(x))`) shared by LFM3 vision and text stacks.
#[derive(Debug, Clone)]
pub struct SwiGluMlp {
    w1: LinearLayer,
    w2: LinearLayer,
    w3: LinearLayer,
}

impl SwiGluMlp {
    /// Loads `w1`, `w2`, and `w3` from `store` under `prefix`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if any linear tensor is missing.
    pub fn load(store: &mut WeightStore, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: store.take_linear(&format!("{prefix}.w1"))?,
            w2: store.take_linear(&format!("{prefix}.w2"))?,
            w3: store.take_linear(&format!("{prefix}.w3"))?,
        })
    }

    /// Evaluates `w2(silu(w1(x)) * w3(x))`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if MLX evaluation fails.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.w1.forward(x)?;
        let silu = ops::multiply(&gate, &ops::sigmoid(&gate)?)?;
        let up = self.w3.forward(x)?;
        let prod = ops::multiply(&silu, &up)?;
        self.w2.forward(&prod)
    }
}

/// Exact Gaussian Error Linear Unit (`0.5 * x * (1 + erf(x / sqrt(2)))`) matching `mlx.nn.gelu`.
///
/// # Errors
/// Returns [`D1Error`] if MLX elementwise evaluation fails.
pub fn gelu_exact(x: &Array) -> Result<Array> {
    let dtype = x.dtype();
    let xf = x.as_dtype(mlx_rs::Dtype::Float32)?;
    let inv_sqrt2 = Array::from_f32(std::f32::consts::FRAC_1_SQRT_2);
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let erf_term = ops::erf(&ops::multiply(&xf, &inv_sqrt2)?)?;
    let out = ops::multiply(&ops::multiply(&half, &xf)?, &ops::add(&one, &erf_term)?)?;
    Ok(out.as_dtype(dtype)?)
}
