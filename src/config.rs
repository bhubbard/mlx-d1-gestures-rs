//! Checkpoint, vision processor, and quantization configuration models for `LiquidAI/d1-2B`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{D1Error, Result};

/// Maximum trained `d1_set_size` recorded by LiquidAI d1 checkpoints.
pub const TRAINED_SET_SIZE: usize = 128;

/// Default vision pixel cap (`1024 * 1024`) when `d1_vision_max_pixels` is absent.
pub const DEFAULT_VISION_MAX_PIXELS: usize = 1_048_576;

/// How bounding-box edge distributions are collapsed to scalar coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxEdgeRead {
    /// Reads the median bin center directly.
    #[default]
    Median,
    /// Linearly interpolates where the cumulative mass crosses `0.5` inside the median bin.
    Interp,
}

/// Layer operator kind inside the LFM3 language or vision stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Causal gated short depthwise convolution (`Lfm3ShortConv` or `VisionConv`).
    Conv,
    /// Full bidirectional/set or causal attention (`Lfm3Attention` or `VisionAttention`).
    FullAttention,
    /// Windowed causal attention (`sliding_window`).
    SlidingAttention,
}

/// Affine weight quantization parameters stored in MLX checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Elements sharing a single scale and bias along the input dimension.
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    /// Bit-width per quantized weight (`4` or `8`).
    #[serde(default = "default_bits")]
    pub bits: i32,
}

const fn default_group_size() -> i32 {
    64
}

const fn default_bits() -> i32 {
    8
}

/// Rotary positional embedding parameters for a specific attention layer family.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    /// Angular frequency base (`rope_theta`).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Fraction of head dimensions rotated (`0.5` leaves half as NoPE).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
}

impl Default for RopeSpec {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Language backbone (`lfm3`) configuration inside `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextConfig {
    /// Vocabulary size (`128000` on `d1-2B`).
    pub vocab_size: usize,
    /// Model hidden dimension (`2048` on `d1-2B`).
    pub hidden_size: usize,
    /// SwiGLU intermediate dimension (`7680` on `d1-2B`).
    pub intermediate_size: usize,
    /// Number of decoder layers (`24` on `d1-2B`).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (`16` on `d1-2B`).
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (`4` on `d1-2B`).
    pub num_key_value_heads: usize,
    /// Per-head dimension (`128` on `d1-2B`).
    pub head_dim: usize,
    /// Causal short-convolution kernel width (`7` on `d1-2B`).
    #[serde(alias = "conv_L_cache", default = "default_conv_l_cache")]
    pub conv_l_cache: usize,
    /// Whether short-convolution projections carry bias terms.
    #[serde(default)]
    pub conv_bias: bool,
    /// RMSNorm epsilon (`1e-5`).
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Per-layer operator schedule (`conv`, `full_attention`, `sliding_attention`).
    pub layer_types: Vec<LayerType>,
    /// RoPE parameters keyed by layer type (`full_attention`, `sliding_attention`).
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeSpec>,
    /// Causal sliding-attention window width (`256`).
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// Whether output token logits share `embed_tokens` weights.
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    /// Whether `post_operator_norm` and `post_ffn_norm` sandwich norms are active.
    #[serde(default = "default_true")]
    pub use_sandwich_norm: bool,
}

impl TextConfig {
    /// Effective causal short-convolution kernel width (`conv_L_cache`).
    #[must_use]
    pub const fn conv_kernel_size(&self) -> usize {
        self.conv_l_cache
    }

    /// RoPE configuration for the given attention layer kind.
    #[must_use]
    pub fn rope_for(&self, layer_type: LayerType) -> RopeSpec {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention | LayerType::Conv => "full_attention",
        };
        self.rope_parameters.get(key).copied().unwrap_or_default()
    }
}

const fn default_conv_l_cache() -> usize {
    7
}

const fn default_norm_eps() -> f32 {
    1e-5
}

const fn default_sliding_window() -> usize {
    256
}

const fn default_true() -> bool {
    true
}

/// Vision encoder (`lfm3_vl_vision`) configuration inside `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Vision hidden dimension (`1024` on `d1-2B`).
    pub hidden_size: usize,
    /// Vision SwiGLU intermediate dimension (`2560` on `d1-2B`).
    pub intermediate_size: usize,
    /// Number of vision layers (`14` on `d1-2B`).
    pub num_hidden_layers: usize,
    /// Number of vision query heads (`16` on `d1-2B`).
    pub num_attention_heads: usize,
    /// Number of vision key/value heads (`8` on `d1-2B`).
    pub num_key_value_heads: usize,
    /// Input RGB channels (`3`).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// Spatial patch size in pixels (`16`).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Spatial merge factor (`2`).
    #[serde(default = "default_merge_size")]
    pub spatial_merge_size: usize,
    /// Temporal patch repetition factor (`2`).
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Depthwise 2D spatial convolution kernel size (`3`).
    #[serde(default = "default_vision_conv_kernel")]
    pub conv_kernel_size: usize,
    /// Vision RMSNorm epsilon (`1e-5`).
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Per-layer schedule (`conv` vs `full_attention`).
    pub layer_types: Vec<LayerType>,
    /// Axial 2D RoPE parameters.
    #[serde(default)]
    pub rope_parameters: RopeSpec,
}

const fn default_in_channels() -> usize {
    3
}

const fn default_patch_size() -> usize {
    16
}

const fn default_merge_size() -> usize {
    2
}

const fn default_temporal_patch_size() -> usize {
    2
}

const fn default_vision_conv_kernel() -> usize {
    3
}

/// Top-level `LiquidAI/d1-2B` / `LiquidAI/d1-2B-MLX-8bit` model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model family identifier (`"lfm3_vl"`).
    pub model_type: String,
    /// Placeholder token ID where projected image features are scattered (`125030`).
    pub image_token_id: u32,
    /// Start-of-vision wrapper token ID (`125009`).
    pub vision_start_token_id: u32,
    /// End-of-vision wrapper token ID (`125010`).
    pub vision_end_token_id: u32,
    /// Spatial downsample factor before the multimodal projector (`2`).
    #[serde(default = "default_merge_size")]
    pub downsample_factor: usize,
    /// Multimodal projector hidden dimension (`4096`).
    pub projector_hidden_size: usize,
    /// Whether multimodal projector linear layers include bias terms (`true`).
    #[serde(default = "default_true")]
    pub projector_bias: bool,
    /// LayerNorm epsilon inside the multimodal projector (`1e-6`).
    #[serde(default = "default_projector_eps")]
    pub projector_layer_norm_eps: f32,
    /// Attention packing mode (`"set"` on d1 checkpoints).
    #[serde(default = "default_d1_attention")]
    pub d1_attention: String,
    /// Option code scheme (`1` or `2`).
    #[serde(default = "default_d1_codes")]
    pub d1_codes: u8,
    /// Whether numeric grids use single-token integer thousandths (`true` on `d1-2B`).
    #[serde(default)]
    pub d1_numbers: bool,
    /// Trained question pack size under `set` attention (`16` on `d1-2B`).
    #[serde(default = "default_d1_set_size")]
    pub d1_set_size: usize,
    /// Maximum pixel budget per input frame (`1638400` on `d1-2B`).
    #[serde(default = "default_vision_max_pixels")]
    pub d1_vision_max_pixels: usize,
    /// Bounding-box edge readout rule (`median` or `interp`).
    #[serde(default)]
    pub d1_box_edges: BoxEdgeRead,
    /// Trained prompt template version (`1` or `2`).
    #[serde(default = "default_d1_prompt")]
    pub d1_prompt: u8,
    /// Optional MLX affine quantization settings (`group_size = 64, bits = 8`).
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// Fallback alias for `quantization_config`.
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    /// LFM3 language backbone configuration.
    pub text_config: TextConfig,
    /// LFM3-VL vision encoder configuration.
    pub vision_config: VisionConfig,
}

const fn default_projector_eps() -> f32 {
    1e-6
}

fn default_d1_attention() -> String {
    "set".to_string()
}

const fn default_d1_codes() -> u8 {
    1
}

const fn default_d1_set_size() -> usize {
    4
}

const fn default_vision_max_pixels() -> usize {
    DEFAULT_VISION_MAX_PIXELS
}

const fn default_d1_prompt() -> u8 {
    1
}

impl ModelConfig {
    /// Loads and validates `config.json` from disk.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the file cannot be read, parsed, or fails d1 invariant checks.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path).map_err(|source| D1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_str(&raw)
    }

    /// Parses and validates `config.json` from a JSON string.
    ///
    /// # Errors
    /// Returns [`D1Error`] if JSON deserialization or invariant validation fails.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Active quantization settings if the checkpoint is quantized.
    #[must_use]
    pub fn effective_quantization(&self) -> Option<QuantizationConfig> {
        self.quantization.or(self.quantization_config)
    }

    /// Validates architectural invariants required by the MLX System-One runner.
    ///
    /// # Errors
    /// Returns [`D1Error::InvalidConfig`] when any hyperparameter is out of bounds.
    pub fn validate(&self) -> Result<()> {
        if self.d1_attention != "set" {
            return Err(D1Error::InvalidConfig(format!(
                "expected d1_attention=\"set\", got {:?}",
                self.d1_attention
            )));
        }
        if !(2..=TRAINED_SET_SIZE).contains(&self.d1_set_size) {
            return Err(D1Error::InvalidConfig(format!(
                "d1_set_size must be in 2..={TRAINED_SET_SIZE}, got {}",
                self.d1_set_size
            )));
        }
        if !matches!(self.d1_prompt, 1 | 2) {
            return Err(D1Error::InvalidConfig(format!(
                "d1_prompt must be 1 or 2, got {}",
                self.d1_prompt
            )));
        }
        if self.text_config.layer_types.len() != self.text_config.num_hidden_layers {
            return Err(D1Error::InvalidConfig(format!(
                "text_config.layer_types length ({}) != num_hidden_layers ({})",
                self.text_config.layer_types.len(),
                self.text_config.num_hidden_layers
            )));
        }
        if self.vision_config.layer_types.len() != self.vision_config.num_hidden_layers {
            return Err(D1Error::InvalidConfig(format!(
                "vision_config.layer_types length ({}) != num_hidden_layers ({})",
                self.vision_config.layer_types.len(),
                self.vision_config.num_hidden_layers
            )));
        }
        Ok(())
    }
}

/// Pixel bounds (`shortest_edge` = min pixels, `longest_edge` = max pixels) in `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelBounds {
    /// Minimum total pixels after `smart_resize`.
    pub shortest_edge: usize,
    /// Maximum total pixels after `smart_resize`.
    pub longest_edge: usize,
}

impl Default for PixelBounds {
    fn default() -> Self {
        Self {
            shortest_edge: 65_536,
            longest_edge: 16_777_216,
        }
    }
}

/// Image preprocessing settings from `processor_config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageProcessorConfig {
    /// Whether to resize frames to multiples of `patch_size * merge_size`.
    #[serde(default = "default_true")]
    pub do_resize: bool,
    /// Whether to scale `u8` pixel intensities by `rescale_factor`.
    #[serde(default = "default_true")]
    pub do_rescale: bool,
    /// Multiplicative factor converting `0..=255` intensities to `[0.0, 1.0]`.
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    /// Whether to normalize RGB channels by `(x - mean) / std`.
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    /// Per-channel mean (`[0.5, 0.5, 0.5]`).
    #[serde(default = "default_half_rgb")]
    pub image_mean: [f32; 3],
    /// Per-channel standard deviation (`[0.5, 0.5, 0.5]`).
    #[serde(default = "default_half_rgb")]
    pub image_std: [f32; 3],
    /// Spatial patch size (`16`).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Spatial merge factor (`2`).
    #[serde(default = "default_merge_size")]
    pub merge_size: usize,
    /// Temporal patch repetition count (`2`).
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Dynamic resize pixel bounds.
    #[serde(default)]
    pub size: PixelBounds,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_rescale_factor(),
            do_normalize: true,
            image_mean: default_half_rgb(),
            image_std: default_half_rgb(),
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            size: PixelBounds::default(),
        }
    }
}

const fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

const fn default_half_rgb() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

/// Wrapper for `processor_config.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Image processor configuration block.
    #[serde(default)]
    pub image_processor: ImageProcessorConfig,
}

impl ProcessorConfig {
    /// Loads `processor_config.json` from disk.
    ///
    /// # Errors
    /// Returns [`D1Error`] if reading or parsing fails.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path).map_err(|source| D1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(serde_json::from_str(&raw)?)
    }
}

/// Subset of `tokenizer_config.json` used to read BOS/special tokens cleanly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMeta {
    /// Beginning-of-sequence token string (`"<|startoftext|>"`).
    #[serde(default = "default_bos_token")]
    pub bos_token: String,
    /// Image placeholder token (`"<|image|>"`).
    #[serde(default = "default_image_token")]
    pub image_token: String,
    /// Vision start wrapper token (`"<|vision_start|>"`).
    #[serde(default = "default_vision_start")]
    pub vision_start_token: String,
    /// Vision end wrapper token (`"<|vision_end|>"`).
    #[serde(default = "default_vision_end")]
    pub vision_end_token: String,
}

impl Default for TokenizerMeta {
    fn default() -> Self {
        Self {
            bos_token: default_bos_token(),
            image_token: default_image_token(),
            vision_start_token: default_vision_start(),
            vision_end_token: default_vision_end(),
        }
    }
}

impl TokenizerMeta {
    /// Loads `tokenizer_config.json` if present, or falls back to LFM3-VL defaults.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the file exists and contains invalid JSON.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path).map_err(|source| D1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(serde_json::from_str(&raw)?)
    }
}

fn default_bos_token() -> String {
    "<|startoftext|>".to_string()
}

fn default_image_token() -> String {
    "<|image|>".to_string()
}

fn default_vision_start() -> String {
    "<|vision_start|>".to_string()
}

fn default_vision_end() -> String {
    "<|vision_end|>".to_string()
}
