//! LFM3-VL vision encoder (`VisionModel`) and GELU `MultiModalProjector` on Apple Silicon MLX.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fast, ops, Array, Dtype};

use crate::config::{LayerType, ModelConfig, VisionConfig};
use crate::error::{D1Error, Result};
use crate::mlx_layers::{gelu_exact, LayerNorm, LinearLayer, RmsNorm, SwiGluMlp, WeightStore};

#[derive(Debug, Clone)]
struct VisionAttention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
}

impl VisionAttention {
    fn load(store: &mut WeightStore, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as i32;
        Ok(Self {
            n_heads: cfg.num_attention_heads as i32,
            n_kv_heads: cfg.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: store.take_linear(&format!("{prefix}.q_proj"))?,
            k_proj: store.take_linear(&format!("{prefix}.k_proj"))?,
            v_proj: store.take_linear(&format!("{prefix}.v_proj"))?,
            out_proj: store.take_linear(&format!("{prefix}.out_proj"))?,
            q_layernorm: store.take_rms_norm(&format!("{prefix}.q_layernorm"), cfg.norm_eps)?,
            k_layernorm: store.take_rms_norm(&format!("{prefix}.k_layernorm"), cfg.norm_eps)?,
        })
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let n = x.shape()[0];
        let q = self
            .q_layernorm
            .forward(
                &self
                    .q_proj
                    .forward(x)?
                    .reshape(&[n, self.n_heads, self.head_dim])?,
            )?;
        let k =
            self.k_layernorm
                .forward(&self.k_proj.forward(x)?.reshape(&[
                    n,
                    self.n_kv_heads,
                    self.head_dim,
                ])?)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[n, self.n_kv_heads, self.head_dim])?;

        let q_rot = apply_axial_rope_3d(&q, cos, sin)?;
        let k_rot = apply_axial_rope_3d(&k, cos, sin)?;

        // [1, H, N, D]
        let q_sdpa =
            q_rot
                .transpose_axes(&[1, 0, 2])?
                .reshape(&[1, self.n_heads, n, self.head_dim])?;
        let k_sdpa =
            k_rot
                .transpose_axes(&[1, 0, 2])?
                .reshape(&[1, self.n_kv_heads, n, self.head_dim])?;
        let v_sdpa =
            v.transpose_axes(&[1, 0, 2])?
                .reshape(&[1, self.n_kv_heads, n, self.head_dim])?;

        let attn =
            fast::scaled_dot_product_attention(&q_sdpa, &k_sdpa, &v_sdpa, self.scale, None, None)?;
        let merged = attn
            .reshape(&[self.n_heads, n, self.head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[n, self.n_heads * self.head_dim])?;
        self.out_proj.forward(&merged)
    }
}

fn apply_axial_rope_3d(t: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let orig_dtype = t.dtype();
    let tf = t.as_dtype(Dtype::Float32)?;
    let parts = ops::split_equal(&tf, 2, -1)?;
    let neg_b = ops::negative(&parts[1])?;
    let rotated_half = ops::concatenate(&[&neg_b, &parts[0]], -1)?;
    let cos_exp = ops::expand_dims(cos, 1)?;
    let sin_exp = ops::expand_dims(sin, 1)?;
    let out = ops::add(
        &ops::multiply(&tf, &cos_exp)?,
        &ops::multiply(&rotated_half, &sin_exp)?,
    )?;
    Ok(out.as_dtype(orig_dtype)?)
}

#[derive(Debug, Clone)]
struct VisionConv {
    merge: i32,
    hidden_size: i32,
    padding: i32,
    in_proj: LinearLayer,
    spatial_conv_weight: Array,
    out_proj: LinearLayer,
}

impl VisionConv {
    fn load(store: &mut WeightStore, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        Ok(Self {
            merge: cfg.spatial_merge_size as i32,
            hidden_size: cfg.hidden_size as i32,
            padding: (cfg.conv_kernel_size / 2) as i32,
            in_proj: store.take_linear(&format!("{prefix}.in_proj"))?,
            spatial_conv_weight: store.take(&format!("{prefix}.spatial_conv.weight"))?,
            out_proj: store.take_linear(&format!("{prefix}.out_proj"))?,
        })
    }

    fn forward(&self, x: &Array, height: i32, width: i32) -> Result<Array> {
        let proj = self.in_proj.forward(x)?;
        let splits = ops::split_equal(&proj, 3, -1)?;
        let b = &splits[0];
        let c = &splits[1];
        let v = &splits[2];
        let m = self.merge;
        let d = self.hidden_size;

        let bv = ops::multiply(b, v)?;
        let grid_in = bv
            .reshape(&[height / m, width / m, m, m, d])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[1, height, width, d])?;

        let conv = ops::conv2d(
            &grid_in,
            &self.spatial_conv_weight,
            (1, 1),
            (self.padding, self.padding),
            (1, 1),
            self.hidden_size,
        )?;

        let unmerged = conv
            .reshape(&[height / m, m, width / m, m, d])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[height * width, d])?;

        let gated = ops::multiply(c, &unmerged)?;
        self.out_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
enum VisionOp {
    Attention(VisionAttention),
    Conv(VisionConv),
}

#[derive(Debug, Clone)]
struct VisionLayer {
    op: VisionOp,
    feed_forward: SwiGluMlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl VisionLayer {
    fn load(store: &mut WeightStore, idx: usize, cfg: &VisionConfig) -> Result<Self> {
        let prefix = format!("vision_tower.layers.{idx}");
        let op = match cfg.layer_types[idx] {
            LayerType::FullAttention | LayerType::SlidingAttention => VisionOp::Attention(
                VisionAttention::load(store, &format!("{prefix}.self_attn"), cfg)?,
            ),
            LayerType::Conv => {
                VisionOp::Conv(VisionConv::load(store, &format!("{prefix}.conv"), cfg)?)
            }
        };
        Ok(Self {
            op,
            feed_forward: SwiGluMlp::load(store, &format!("{prefix}.feed_forward"))?,
            operator_norm: store.take_rms_norm(&format!("{prefix}.operator_norm"), cfg.norm_eps)?,
            ffn_norm: store.take_rms_norm(&format!("{prefix}.ffn_norm"), cfg.norm_eps)?,
        })
    }

    fn forward(
        &self,
        x: &Array,
        height: i32,
        width: i32,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let normed = self.operator_norm.forward(x)?;
        let delta = match &self.op {
            VisionOp::Attention(attn) => attn.forward(&normed, cos, sin)?,
            VisionOp::Conv(conv) => conv.forward(&normed, height, width)?,
        };
        let x1 = ops::add(x, &delta)?;
        let ffn_out = self.feed_forward.forward(&self.ffn_norm.forward(&x1)?)?;
        Ok(ops::add(&x1, &ffn_out)?)
    }
}

/// LFM3-VL vision encoder (`vision_tower.*`).
#[derive(Debug, Clone)]
pub struct VisionTower {
    spatial_merge_size: usize,
    spatial_dim: usize,
    rope_theta: f32,
    patchifier: LinearLayer,
    embedding_norm: RmsNorm,
    layers: Vec<VisionLayer>,
}

impl VisionTower {
    /// Loads `vision_tower.*` weights from `store`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if any vision layer tensor is missing.
    pub fn load(store: &mut WeightStore, cfg: &VisionConfig) -> Result<Self> {
        let patchifier = store.take_linear("vision_tower.patchifier")?;
        let embedding_norm = store.take_rms_norm("vision_tower.embedding_norm", cfg.norm_eps)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionLayer::load(store, i, cfg)?);
        }
        Ok(Self {
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_dim: cfg.hidden_size / cfg.num_attention_heads / 2,
            rope_theta: cfg.rope_parameters.rope_theta,
            patchifier,
            embedding_norm,
            layers,
        })
    }

    /// Encodes flattened patches `[N_patches, patch_dim]` for a single `[temporal, height, width]` grid.
    ///
    /// # Errors
    /// Returns [`D1Error`] if patch count mismatches `grid_thw` or MLX kernels fail.
    pub fn forward(&self, pixel_values: &Array, grid_thw: [usize; 3]) -> Result<Array> {
        let [temporal, height, width] = grid_thw;
        let expected = temporal * height * width;
        if pixel_values.shape()[0] as usize != expected {
            return Err(D1Error::InvalidRequest(format!(
                "pixel_values rows ({}) != grid_thw product ({expected})",
                pixel_values.shape()[0]
            )));
        }

        let (cos, sin) = build_vision_axial_cos_sin(
            height,
            width,
            self.spatial_merge_size,
            self.spatial_dim,
            self.rope_theta,
        );

        let target_dtype = self.embedding_norm.weight.dtype();
        let patches_per_frame = (height * width) as i32;
        let h_i32 = height as i32;
        let w_i32 = width as i32;

        let mut frame_outputs = Vec::with_capacity(temporal);
        for t in 0..temporal {
            let start = (t as i32) * patches_per_frame;
            let stop = start + patches_per_frame;
            let slice = pixel_values
                .index((start..stop, ..))
                .as_dtype(target_dtype)?;
            let mut x = self
                .embedding_norm
                .forward(&self.patchifier.forward(&slice)?)?;
            for layer in &self.layers {
                x = layer.forward(&x, h_i32, w_i32, &cos, &sin)?;
            }
            frame_outputs.push(x);
        }

        if frame_outputs.len() == 1 {
            Ok(frame_outputs.remove(0))
        } else {
            let refs: Vec<&Array> = frame_outputs.iter().collect();
            Ok(ops::concatenate(&refs, 0)?)
        }
    }
}

fn build_vision_axial_cos_sin(
    height: usize,
    width: usize,
    merge: usize,
    spatial_dim: usize,
    rope_theta: f32,
) -> (Array, Array) {
    let half_pairs = spatial_dim / 2;
    let inv_freq: Vec<f32> = (0..half_pairs)
        .map(|k| rope_theta.powf(-((2 * k) as f32) / (spatial_dim as f32)))
        .collect();

    let n_tokens = height * width;
    let head_dim = spatial_dim * 2;
    let mut cos_vec = Vec::with_capacity(n_tokens * head_dim);
    let mut sin_vec = Vec::with_capacity(n_tokens * head_dim);

    // Matches `grid_positions`: (height // merge, merge, width // merge, merge, 2).transpose(0, 2, 1, 3, 4)
    for bh in 0..(height / merge) {
        for bw in 0..(width / merge) {
            for mh in 0..merge {
                for mw in 0..merge {
                    let row = (bh * merge + mh) as f32;
                    let col = (bw * merge + mw) as f32;
                    let mut half_angles = Vec::with_capacity(spatial_dim);
                    for &f in &inv_freq {
                        half_angles.push(row * f);
                    }
                    for &f in &inv_freq {
                        half_angles.push(col * f);
                    }
                    // Repeated twice (`concatenate([angles, angles], axis=-1)`)
                    for _ in 0..2 {
                        for &ang in &half_angles {
                            cos_vec.push(ang.cos());
                            sin_vec.push(ang.sin());
                        }
                    }
                }
            }
        }
    }

    let shape = [n_tokens as i32, head_dim as i32];
    (
        Array::from_slice(&cos_vec, &shape),
        Array::from_slice(&sin_vec, &shape),
    )
}

/// Downsampling `LayerNorm -> Flatten(downsample^2) -> Linear -> GELU -> Linear` projector.
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    input_dim: i32,
    layer_norm: LayerNorm,
    linear_1: LinearLayer,
    linear_2: LinearLayer,
}

impl MultiModalProjector {
    /// Loads `multi_modal_projector.*` weights from `store`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if any projector tensor is missing.
    pub fn load(store: &mut WeightStore, cfg: &ModelConfig) -> Result<Self> {
        let input_dim =
            (cfg.vision_config.hidden_size * cfg.downsample_factor * cfg.downsample_factor) as i32;
        Ok(Self {
            input_dim,
            layer_norm: store.take_layer_norm(
                "multi_modal_projector.layer_norm",
                cfg.projector_layer_norm_eps,
            )?,
            linear_1: store.take_linear("multi_modal_projector.linear_1")?,
            linear_2: store.take_linear("multi_modal_projector.linear_2")?,
        })
    }

    /// Projects `[N_patches, vision_hidden]` into `[N_patches / downsample^2, text_hidden]`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if MLX operations fail.
    pub fn forward(&self, vision_hidden: &Array) -> Result<Array> {
        let normed = self
            .layer_norm
            .forward(vision_hidden)?
            .reshape(&[-1, self.input_dim])?;
        let hidden = gelu_exact(&self.linear_1.forward(&normed)?)?;
        self.linear_2.forward(&hidden)
    }
}
