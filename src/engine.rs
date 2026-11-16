//! End-to-end `LiquidAI/d1-2B` System-One multimodal decision engine on Apple Silicon MLX.

use std::collections::HashMap;
use std::time::Instant;

use image::{DynamicImage, Rgb, RgbImage};
use indexmap::IndexMap;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use serde::{Deserialize, Serialize};

use crate::calibration::Calibration;
use crate::config::{ModelConfig, ProcessorConfig};
use crate::error::{D1Error, Result};
use crate::hub::resolve_checkpoint_dir;
use crate::image_proc::preprocess_image;
use crate::language::{Lfm3Backbone, SetMasks};
use crate::mlx_layers::WeightStore;
use crate::prompt::{slot_probabilities, PackedGroup, PromptBuilder, SlotReadout};
use crate::questions::{build_answer, Answer, Question, ScoreReadout};
use crate::vision::{MultiModalProjector, VisionTower};

/// Token accounting reported in every System-One decision response (`output_tokens` is always `0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionUsage {
    /// Total packed prefill tokens read across the forward pass(es).
    pub input_tokens: usize,
    /// Generated autoregressive tokens (always `0` for System-One decision models).
    pub output_tokens: usize,
}

/// Complete JSON payload returned by `POST /run` and CLI `decide`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// Named calibrated answers matching the caller's `questions` keys.
    pub answers: IndexMap<String, Answer>,
    /// Prefill token usage (`output_tokens: 0`).
    pub usage: DecisionUsage,
    /// Wall-clock inference latency in milliseconds (rounded to `0.1` ms).
    pub ms: f64,
}

#[derive(Debug, Clone)]
struct PendingSlotEntry {
    q_local: usize,
    slot_local: usize,
    abs_pos: usize,
    readout: SlotReadout,
}

#[derive(Debug, Clone)]
struct BatchedTableGroup {
    slot_indices_arr: Array,
    candidate_ids_arr: Array,
    num_candidates: usize,
    targets: Vec<(usize, usize, SlotReadout)>, // (question_idx_in_group, slot_idx_in_question, readout)
}

#[derive(Debug, Clone)]
struct CompiledGroupPlan {
    packed: PackedGroup,
    masks: SetMasks,
    table_batches: Vec<BatchedTableGroup>,
    slot_counts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    num_image_tokens: usize,
    state: Option<String>,
    questions_json: String,
    set_size: usize,
}

/// Native Apple Silicon MLX runner for `LiquidAI/d1-2B` / `LiquidAI/d1-2B-MLX-8bit`.
#[derive(Debug)]
pub struct D1Engine {
    model_id: String,
    config: ModelConfig,
    processor_config: ProcessorConfig,
    prompt_builder: PromptBuilder,
    calibration: Option<Calibration>,
    vision_tower: VisionTower,
    projector: MultiModalProjector,
    backbone: Lfm3Backbone,
    plan_cache: HashMap<PlanCacheKey, Vec<CompiledGroupPlan>>,
}

impl D1Engine {
    /// Resolves (or downloads) `model_spec` and loads all weights onto Apple Silicon unified memory.
    ///
    /// # Errors
    /// Returns [`D1Error`] if Hub resolution, JSON parsing, or SafeTensors weight loading fails.
    pub fn load(model_spec: &str) -> Result<Self> {
        let checkpoint_dir = resolve_checkpoint_dir(model_spec)?;
        let config = ModelConfig::load(checkpoint_dir.join("config.json"))?;
        let processor_path = checkpoint_dir.join("processor_config.json");
        let processor_config = if processor_path.is_file() {
            ProcessorConfig::load(processor_path)?
        } else {
            ProcessorConfig::default()
        };
        let calibration = Calibration::load_optional(checkpoint_dir.join("calibration.json"))?;
        let prompt_builder = PromptBuilder::from_checkpoint_dir(&checkpoint_dir, &config)?;

        let mut store = WeightStore::load_from_dir(
            &checkpoint_dir,
            config.effective_quantization(),
            config.text_config.tie_word_embeddings,
        )?;

        Self::from_parts(
            model_spec.to_string(),
            config,
            processor_config,
            prompt_builder,
            calibration,
            &mut store,
        )
    }

    /// Builds a [`D1Engine`] from already-parsed configs, tokenizer, and [`WeightStore`].
    ///
    /// # Errors
    /// Returns [`D1Error`] if any required model tensor is missing from `store`.
    pub fn from_parts(
        model_id: String,
        config: ModelConfig,
        processor_config: ProcessorConfig,
        prompt_builder: PromptBuilder,
        calibration: Option<Calibration>,
        store: &mut WeightStore,
    ) -> Result<Self> {
        config.validate()?;
        let vision_tower = VisionTower::load(store, &config.vision_config)?;
        let projector = MultiModalProjector::load(store, &config)?;
        let backbone = Lfm3Backbone::load(store, &config.text_config)?;

        Ok(Self {
            model_id,
            config,
            processor_config,
            prompt_builder,
            calibration,
            vision_tower,
            projector,
            backbone,
            plan_cache: HashMap::new(),
        })
    }

    /// Model identifier string (`LiquidAI/d1-2B-MLX-8bit` or local path).
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Loaded [`ModelConfig`].
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Warms up Metal shader compilation on a `640x480` neutral gray RGB frame (`decide.py` parity).
    ///
    /// # Errors
    /// Returns [`D1Error`] if the warm-up forward pass fails.
    pub fn warmup(&mut self, questions: &IndexMap<String, Question>) -> Result<DecisionResponse> {
        let gray = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 480, Rgb([128, 128, 128])));
        self.decide_image(&gray, None, questions, ScoreReadout::Median)
    }

    /// Decodes an encoded JPEG/PNG/WebP byte buffer and evaluates `questions` in one System-One pass.
    ///
    /// # Errors
    /// Returns [`D1Error`] if image decoding or model inference fails.
    pub fn decide_encoded_frame(
        &mut self,
        image_bytes: &[u8],
        questions: &IndexMap<String, Question>,
    ) -> Result<DecisionResponse> {
        let img = image::load_from_memory(image_bytes)?;
        self.decide_image(&img, None, questions, ScoreReadout::Median)
    }

    /// Evaluates named `questions` over an optional [`DynamicImage`] and optional text `state`.
    ///
    /// # Errors
    /// Returns [`D1Error`] if preprocessing, tokenization, or Metal execution fails.
    pub fn decide_image(
        &mut self,
        image: &DynamicImage,
        state: Option<&str>,
        questions: &IndexMap<String, Question>,
        score_readout: ScoreReadout,
    ) -> Result<DecisionResponse> {
        let force_cpu = std::env::var("DECIDE_DEVICE")
            .map(|v| v.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        if force_cpu {
            mlx_rs::with_device(mlx_rs::Device::cpu(), || {
                self.decide_image_inner(image, state, questions, score_readout)
            })
        } else {
            self.decide_image_inner(image, state, questions, score_readout)
        }
    }

    fn decide_image_inner(
        &mut self,
        image: &DynamicImage,
        state: Option<&str>,
        questions: &IndexMap<String, Question>,
        score_readout: ScoreReadout,
    ) -> Result<DecisionResponse> {
        let t0 = Instant::now();

        let max_pixels = std::env::var("D1_VISION_MAX_PIXELS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(self.config.d1_vision_max_pixels);

        let processed =
            preprocess_image(image, &self.processor_config.image_processor, max_pixels)?;

        let pixel_arr = Array::from_slice(
            &processed.pixel_values,
            &[processed.num_patches as i32, processed.patch_dim as i32],
        );
        let vision_hidden = self.vision_tower.forward(&pixel_arr, processed.grid_thw)?;
        let image_features = self.projector.forward(&vision_hidden)?;
        image_features.eval()?;

        let q_vec: Vec<Question> = questions.values().cloned().collect();
        let plans = self.get_or_compile_plans(
            processed.num_image_tokens,
            state,
            &q_vec,
            self.config.d1_set_size,
        )?;

        let mut total_input_tokens = 0usize;
        let mut all_question_probs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(q_vec.len());

        for plan in &plans {
            total_input_tokens += plan.packed.input_ids.len();
            let hidden = self.backbone.forward_hidden(
                &plan.packed.input_ids,
                &plan.masks,
                self.config.image_token_id,
                Some(&image_features),
            )?;

            let mut group_probs: Vec<Vec<Vec<f64>>> = plan
                .slot_counts
                .iter()
                .map(|&n| vec![Vec::new(); n])
                .collect();

            let mut pending_logits = Vec::with_capacity(plan.table_batches.len());
            for batch in &plan.table_batches {
                let slot_h = hidden.index(&batch.slot_indices_arr);
                let logits_f32 = self
                    .backbone
                    .project_slot_candidates(&slot_h, &batch.candidate_ids_arr)?
                    .as_dtype(Dtype::Float32)?;
                logits_f32.eval()?;
                pending_logits.push((batch, logits_f32));
            }

            for (batch, logits_arr) in pending_logits {
                let slice = logits_arr
                    .try_as_slice::<f32>()
                    .map_err(|e| D1Error::Mlx(format!("reading candidate logits slice: {e}")))?;
                let stride = batch.num_candidates;
                for (row_idx, (q_local, slot_local, readout)) in batch.targets.iter().enumerate() {
                    let row = &slice[row_idx * stride..(row_idx + 1) * stride];
                    let probs = slot_probabilities(readout, row, self.calibration.as_ref());
                    group_probs[*q_local][*slot_local] = probs;
                }
            }

            all_question_probs.extend(group_probs);
        }

        let mut answers = IndexMap::with_capacity(questions.len());
        for ((name, q), slot_probs) in questions.iter().zip(all_question_probs.iter()) {
            let ans = build_answer(
                q,
                slot_probs,
                score_readout,
                self.calibration.as_ref(),
                self.config.d1_box_edges,
            )?;
            answers.insert(name.clone(), ans);
        }

        let elapsed_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
        Ok(DecisionResponse {
            answers,
            usage: DecisionUsage {
                input_tokens: total_input_tokens,
                output_tokens: 0,
            },
            ms: elapsed_ms,
        })
    }

    fn get_or_compile_plans(
        &mut self,
        num_image_tokens: usize,
        state: Option<&str>,
        questions: &[Question],
        set_size: usize,
    ) -> Result<Vec<CompiledGroupPlan>> {
        let key = PlanCacheKey {
            num_image_tokens,
            state: state.map(ToString::to_string),
            questions_json: serde_json::to_string(questions)?,
            set_size,
        };
        if let Some(cached) = self.plan_cache.get(&key) {
            return Ok(cached.clone());
        }

        let packed_groups =
            self.prompt_builder
                .pack_questions(num_image_tokens, state, questions, set_size)?;
        let mut compiled = Vec::with_capacity(packed_groups.len());

        for packed in packed_groups {
            let masks = SetMasks::build(
                packed.input_ids.len(),
                packed.state_len,
                &packed.question_groups,
                self.backbone.sliding_window(),
            );
            let slot_counts: Vec<usize> = packed.question_slots.iter().map(Vec::len).collect();

            // Group slots that share identical `unique_tokens` (e.g. all 8 box edges share 1,000 numbers)
            let mut by_tokens: IndexMap<Vec<u32>, Vec<PendingSlotEntry>> = IndexMap::new();
            for (q_local, slots) in packed.question_slots.iter().enumerate() {
                for (slot_local, (abs_pos, readout)) in slots.iter().enumerate() {
                    by_tokens
                        .entry(readout.unique_tokens.clone())
                        .or_default()
                        .push(PendingSlotEntry {
                            q_local,
                            slot_local,
                            abs_pos: *abs_pos,
                            readout: readout.clone(),
                        });
                }
            }

            let mut table_batches = Vec::with_capacity(by_tokens.len());
            for (tokens, entries) in by_tokens {
                let slot_positions: Vec<i32> = entries.iter().map(|e| e.abs_pos as i32).collect();
                let cand_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                let slot_indices_arr =
                    Array::from_slice(&slot_positions, &[slot_positions.len() as i32]);
                let candidate_ids_arr = Array::from_slice(&cand_i32, &[cand_i32.len() as i32]);
                let num_candidates = tokens.len();
                let targets = entries
                    .into_iter()
                    .map(|e| (e.q_local, e.slot_local, e.readout))
                    .collect();

                table_batches.push(BatchedTableGroup {
                    slot_indices_arr,
                    candidate_ids_arr,
                    num_candidates,
                    targets,
                });
            }

            compiled.push(CompiledGroupPlan {
                packed,
                masks,
                table_batches,
                slot_counts,
            });
        }

        if self.plan_cache.len() >= 128 {
            self.plan_cache.clear();
        }
        self.plan_cache.insert(key, compiled.clone());
        Ok(compiled)
    }
}
