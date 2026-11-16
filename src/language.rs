//! LFM3 hybrid ShortConv + GQA Attention language backbone with `set`-attention masks.

use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fast, ops, Array, Dtype};

use crate::config::{LayerType, RopeSpec, TextConfig};
use crate::error::{D1Error, Result};
use crate::mlx_layers::{EmbeddingLayer, LinearLayer, RmsNorm, SwiGluMlp, WeightStore};

/// Precomputed boolean attention masks for one packed sequence under `d1` `set` attention.
#[derive(Debug, Clone)]
pub struct SetMasks {
    /// Bidirectional over state + intra-question mask `[T, T]` for `full_attention` layers.
    pub full_mask: Array,
    /// Optional explicit windowed causal mask `[T, T]` when `T > sliding_window` (`None` uses fast `Causal`).
    pub sliding_mask: Option<Array>,
}

impl SetMasks {
    /// Builds `full_attention` and `sliding_attention` masks for a packed sequence of `length` tokens.
    #[must_use]
    pub fn build(
        length: usize,
        state_len: usize,
        question_groups: &[i32],
        sliding_window: usize,
    ) -> Self {
        let mut full_bools = Vec::with_capacity(length * length);
        for (i, &gi) in question_groups.iter().enumerate().take(length) {
            for (j, &gj) in question_groups.iter().enumerate().take(length) {
                let allow = i < state_len || j < state_len || gi == gj;
                full_bools.push(allow);
            }
        }
        let shape = [length as i32, length as i32];
        let full_mask = Array::from_slice(&full_bools, &shape);

        let sliding_mask = if length > sliding_window {
            let mut slide_bools = Vec::with_capacity(length * length);
            for i in 0..length {
                for j in 0..length {
                    let allow = i >= j && i < j + sliding_window;
                    slide_bools.push(allow);
                }
            }
            Some(Array::from_slice(&slide_bools, &shape))
        } else {
            None
        };

        Self {
            full_mask,
            sliding_mask,
        }
    }
}

/// Partial rotary positional embedding rotating matching pairs in both head halves (leaving NoPE intact).
#[derive(Debug, Clone)]
struct PartialRotaryEmbedding {
    dim: usize,
    angles: usize,
    base: f32,
}

impl PartialRotaryEmbedding {
    fn new(dim: usize, spec: RopeSpec) -> Result<Self> {
        let angles = ((dim as f32) * spec.partial_rotary_factor) as usize / 2;
        if angles == 0 || angles > dim / 2 {
            return Err(D1Error::InvalidConfig(format!(
                "invalid partial_rotary_factor {} for head_dim {dim}",
                spec.partial_rotary_factor
            )));
        }
        Ok(Self {
            dim,
            angles,
            base: spec.rope_theta,
        })
    }

    fn build_cos_sin(&self, seq_len: usize, dtype: Dtype) -> Result<(Array, Array)> {
        let half = self.dim / 2;
        let mut inv_freq = vec![0.0_f32; half];
        for (k, slot) in inv_freq.iter_mut().enumerate().take(self.angles) {
            *slot = self.base.powf(-((2 * k) as f32) / (self.dim as f32));
        }

        let mut cos_vec = Vec::with_capacity(seq_len * self.dim);
        let mut sin_vec = Vec::with_capacity(seq_len * self.dim);
        for pos in 0..seq_len {
            let p = pos as f32;
            for _ in 0..2 {
                for &freq in &inv_freq {
                    let ang = p * freq;
                    cos_vec.push(ang.cos());
                    sin_vec.push(ang.sin());
                }
            }
        }
        let shape = [1, 1, seq_len as i32, self.dim as i32];
        let cos = Array::from_slice(&cos_vec, &shape).as_dtype(dtype)?;
        let sin = Array::from_slice(&sin_vec, &shape).as_dtype(dtype)?;
        Ok((cos, sin))
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let parts = ops::split_equal(x, 2, -1)?;
        let neg_b = ops::negative(&parts[1])?;
        let rotated = ops::concatenate(&[&neg_b, &parts[0]], -1)?;
        Ok(ops::add(
            &ops::multiply(x, cos)?,
            &ops::multiply(&rotated, sin)?,
        )?)
    }
}

#[derive(Debug, Clone)]
struct Lfm3Attention {
    layer_type: LayerType,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate_proj: LinearLayer,
    rope: PartialRotaryEmbedding,
}

impl Lfm3Attention {
    fn load(
        store: &mut WeightStore,
        prefix: &str,
        cfg: &TextConfig,
        layer_type: LayerType,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim as i32;
        Ok(Self {
            layer_type,
            n_heads: cfg.num_attention_heads as i32,
            n_kv_heads: cfg.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: store.take_linear(&format!("{prefix}.q_proj"))?,
            k_proj: store.take_linear(&format!("{prefix}.k_proj"))?,
            v_proj: store.take_linear(&format!("{prefix}.v_proj"))?,
            o_proj: store.take_linear(&format!("{prefix}.o_proj"))?,
            q_norm: store.take_rms_norm(&format!("{prefix}.q_norm"), cfg.norm_eps)?,
            k_norm: store.take_rms_norm(&format!("{prefix}.k_norm"), cfg.norm_eps)?,
            gate_proj: store.take_linear(&format!("{prefix}.gate_proj"))?,
            rope: PartialRotaryEmbedding::new(cfg.head_dim, cfg.rope_for(layer_type))?,
        })
    }

    fn forward(&self, x: &Array, masks: &SetMasks) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let length = shape[1];

        let q = self
            .q_norm
            .forward(&self.q_proj.forward(x)?.reshape(&[
                batch,
                length,
                self.n_heads,
                self.head_dim,
            ])?)?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k = self
            .k_norm
            .forward(&self.k_proj.forward(x)?.reshape(&[
                batch,
                length,
                self.n_kv_heads,
                self.head_dim,
            ])?)?
            .transpose_axes(&[0, 2, 1, 3])?;

        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, length, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (cos, sin) = self.rope.build_cos_sin(length as usize, q.dtype())?;
        let q_rot = self.rope.forward(&q, &cos, &sin)?;
        let k_rot = self.rope.forward(&k, &cos, &sin)?;

        let mask_arg = match self.layer_type {
            LayerType::FullAttention | LayerType::Conv => {
                Some(ScaledDotProductAttentionMask::Array(&masks.full_mask))
            }
            LayerType::SlidingAttention => match &masks.sliding_mask {
                Some(arr) => Some(ScaledDotProductAttentionMask::Array(arr)),
                None => Some(ScaledDotProductAttentionMask::Causal),
            },
        };

        let attn =
            fast::scaled_dot_product_attention(&q_rot, &k_rot, &v, self.scale, mask_arg, None)?;
        let attn_t = attn.transpose_axes(&[0, 2, 1, 3])?;

        let gate = ops::expand_dims(&ops::sigmoid(&self.gate_proj.forward(x)?)?, -1)?;
        let gated = ops::multiply(&attn_t, &gate)?.reshape(&[
            batch,
            length,
            self.n_heads * self.head_dim,
        ])?;
        self.o_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct ShortConv {
    l_cache: i32,
    hidden_size: i32,
    conv_weight: Array,
    conv_bias: Option<Array>,
    in_proj: LinearLayer,
    out_proj: LinearLayer,
}

impl ShortConv {
    fn load(store: &mut WeightStore, prefix: &str, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            l_cache: cfg.conv_kernel_size() as i32,
            hidden_size: cfg.hidden_size as i32,
            conv_weight: store.take(&format!("{prefix}.conv.weight"))?,
            conv_bias: store.take_optional(&format!("{prefix}.conv.bias")),
            in_proj: store.take_linear(&format!("{prefix}.in_proj"))?,
            out_proj: store.take_linear(&format!("{prefix}.out_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let bcx = self.in_proj.forward(x)?;
        let splits = ops::split_equal(&bcx, 3, -1)?;
        let b = &splits[0];
        let c = &splits[1];
        let x_part = &splits[2];

        let bx = ops::multiply(b, x_part)?;
        let padded = if self.l_cache > 1 {
            let batch = bx.shape()[0];
            let zeros = Array::zeros::<f32>(&[batch, self.l_cache - 1, self.hidden_size])?
                .as_dtype(bx.dtype())?;
            ops::concatenate(&[&zeros, &bx], 1)?
        } else {
            bx
        };

        let mut conv_out = ops::conv1d(
            &padded.as_dtype(self.conv_weight.dtype())?,
            &self.conv_weight,
            1,
            0,
            1,
            self.hidden_size,
        )?
        .as_dtype(x.dtype())?;

        if let Some(bias) = &self.conv_bias {
            conv_out = ops::add(&conv_out, bias)?;
        }

        let gated = ops::multiply(c, &conv_out)?;
        self.out_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
enum DecoderOp {
    Attention(Lfm3Attention),
    Conv(ShortConv),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    op: DecoderOp,
    feed_forward: SwiGluMlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_operator_norm: Option<RmsNorm>,
    post_ffn_norm: Option<RmsNorm>,
}

impl DecoderLayer {
    fn load(store: &mut WeightStore, idx: usize, cfg: &TextConfig) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{idx}");
        let layer_type = cfg.layer_types[idx];
        let op = match layer_type {
            LayerType::Conv => {
                DecoderOp::Conv(ShortConv::load(store, &format!("{prefix}.conv"), cfg)?)
            }
            LayerType::FullAttention | LayerType::SlidingAttention => DecoderOp::Attention(
                Lfm3Attention::load(store, &format!("{prefix}.self_attn"), cfg, layer_type)?,
            ),
        };

        let (post_operator_norm, post_ffn_norm) = if cfg.use_sandwich_norm {
            (
                Some(store.take_rms_norm(&format!("{prefix}.post_operator_norm"), cfg.norm_eps)?),
                Some(store.take_rms_norm(&format!("{prefix}.post_ffn_norm"), cfg.norm_eps)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            op,
            feed_forward: SwiGluMlp::load(store, &format!("{prefix}.feed_forward"))?,
            operator_norm: store.take_rms_norm(&format!("{prefix}.operator_norm"), cfg.norm_eps)?,
            ffn_norm: store.take_rms_norm(&format!("{prefix}.ffn_norm"), cfg.norm_eps)?,
            post_operator_norm,
            post_ffn_norm,
        })
    }

    fn forward(&self, x: &Array, masks: &SetMasks) -> Result<Array> {
        let normed = self.operator_norm.forward(x)?;
        let mut h = match &self.op {
            DecoderOp::Attention(attn) => attn.forward(&normed, masks)?,
            DecoderOp::Conv(conv) => conv.forward(&normed)?,
        };
        if let Some(post) = &self.post_operator_norm {
            h = post.forward(&h)?;
        }
        let x1 = ops::add(x, &h)?;

        let mut ffn_out = self.feed_forward.forward(&self.ffn_norm.forward(&x1)?)?;
        if let Some(post) = &self.post_ffn_norm {
            ffn_out = post.forward(&ffn_out)?;
        }
        Ok(ops::add(&x1, &ffn_out)?)
    }
}

/// LFM3 language model backbone (`language_model.model.*`) with tied or untied output head.
#[derive(Debug, Clone)]
pub struct Lfm3Backbone {
    sliding_window: usize,
    embed_tokens: EmbeddingLayer,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Option<LinearLayer>,
}

impl Lfm3Backbone {
    /// Loads `language_model.*` weights from `store`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if any required decoder weight tensor is missing.
    pub fn load(store: &mut WeightStore, cfg: &TextConfig) -> Result<Self> {
        let embed_tokens = store.take_embedding("language_model.model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(store, i, cfg)?);
        }
        let embedding_norm =
            store.take_rms_norm("language_model.model.embedding_norm", cfg.norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(store.take_linear("language_model.lm_head")?)
        };

        Ok(Self {
            sliding_window: cfg.sliding_window,
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
        })
    }

    /// Configured causal sliding window size.
    #[must_use]
    pub const fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    /// Runs one packed row `[T]` under `masks`, scattering `image_features` at `image_token_id` positions.
    ///
    /// Returns final normalized hidden states `[T, D]`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if placeholder counts mismatch `image_features` or MLX ops fail.
    pub fn forward_hidden(
        &self,
        ids: &[u32],
        masks: &SetMasks,
        image_token_id: u32,
        image_features: Option<&Array>,
    ) -> Result<Array> {
        let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let token_arr = Array::from_slice(&ids_i32, &[ids.len() as i32]);
        let embedded = self.embed_tokens.embed(&token_arr)?;

        let merged = match image_features {
            None => embedded,
            Some(feats) => splice_image_features(&embedded, ids, image_token_id, feats)?,
        };

        let seq_len = ids.len() as i32;
        let hidden_dim = merged.shape()[1];
        let mut h = merged.reshape(&[1, seq_len, hidden_dim])?;
        for layer in &self.layers {
            h = layer.forward(&h, masks)?;
        }
        let normed = self.embedding_norm.forward(&h)?;
        Ok(normed.reshape(&[seq_len, hidden_dim])?)
    }

    /// Projects slot hidden states `[S, D]` strictly onto `candidate_ids` `[K]` -> `[S, K]`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if candidate projection fails.
    pub fn project_slot_candidates(
        &self,
        slot_hidden: &Array,
        candidate_ids: &Array,
    ) -> Result<Array> {
        match &self.lm_head {
            None => self
                .embed_tokens
                .project_candidates(slot_hidden, candidate_ids),
            Some(head) => {
                let full = head.forward(slot_hidden)?;
                Ok(full.index((.., candidate_ids)))
            }
        }
    }
}

fn splice_image_features(
    embedded: &Array,
    ids: &[u32],
    image_token_id: u32,
    features: &Array,
) -> Result<Array> {
    let placeholder_count = ids.iter().filter(|&&t| t == image_token_id).count();
    let feat_rows = features.shape()[0] as usize;
    if placeholder_count != feat_rows {
        return Err(D1Error::InvalidRequest(format!(
            "image placeholder count ({placeholder_count}) != encoded visual features ({feat_rows})"
        )));
    }
    if placeholder_count == 0 {
        return Ok(embedded.clone());
    }

    let feats_cast = features.as_dtype(embedded.dtype())?;

    let mut segments: Vec<Array> = Vec::new();
    let mut tok_cursor = 0usize;
    let mut feat_cursor = 0usize;

    while tok_cursor < ids.len() {
        if ids[tok_cursor] == image_token_id {
            let run_start = tok_cursor;
            while tok_cursor < ids.len() && ids[tok_cursor] == image_token_id {
                tok_cursor += 1;
            }
            let run_len = tok_cursor - run_start;
            let f_slice =
                feats_cast.index(((feat_cursor as i32)..((feat_cursor + run_len) as i32), ..));
            segments.push(f_slice);
            feat_cursor += run_len;
        } else {
            let text_start = tok_cursor;
            while tok_cursor < ids.len() && ids[tok_cursor] != image_token_id {
                tok_cursor += 1;
            }
            let t_slice = embedded.index(((text_start as i32)..(tok_cursor as i32), ..));
            segments.push(t_slice);
        }
    }

    let refs: Vec<&Array> = segments.iter().collect();
    Ok(ops::concatenate(&refs, 0)?)
}
