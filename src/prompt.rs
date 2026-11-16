//! System-One ChatML prompt construction, option code verbalizer, and `set`-attention slot packing.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use tokenizers::Tokenizer;

use crate::box_decode::EDGE_NAMES;
use crate::calibration::{Calibration, CalibrationKind};
use crate::config::{ModelConfig, TokenizerMeta};
use crate::error::{D1Error, Result};
use crate::grid::{Grid, GRID_128, THOUSANDTHS_GRID};
use crate::questions::Question;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const LFM3_LEAD: &str = "<think></think>";
const ZERO_WIDTH_BREAK: &str = "\u{200b}";
const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
const NO_FORMS: [&str; 3] = ["no", "No", "NO"];

/// Slot readout specification for a single answer token position.
#[derive(Debug, Clone)]
pub struct SlotReadout {
    /// Flattened unique vocabulary token IDs gathered for this slot.
    pub unique_tokens: Vec<u32>,
    /// Per-option index lists into `unique_tokens` (max-pooled per option).
    pub option_forms: Vec<Vec<usize>>,
    /// Calibration bucket when temperature scaling applies (`None` for typed slots).
    pub calibration_kind: Option<CalibrationKind>,
}

/// Packed prompt sequence and slot metadata for a group of questions under `set` attention.
#[derive(Debug, Clone)]
pub struct PackedGroup {
    /// Full sequence of token IDs (`prefix` + packed question suffixes).
    pub input_ids: Vec<u32>,
    /// Number of prefix/state tokens before the first question.
    pub state_len: usize,
    /// Per-token group assignment (`-1` on state tokens, `1..=G` on each question's tokens).
    pub question_groups: Vec<i32>,
    /// Per-question list of `(token_position, SlotReadout)` pairs in question order.
    pub question_slots: Vec<Vec<(usize, SlotReadout)>>,
}

/// Tokenizer wrapper precomputing special-token escape rules and thousandth-number readout tables.
pub struct PromptBuilder {
    tokenizer: Tokenizer,
    meta: TokenizerMeta,
    escape_regex: Option<Regex>,
    d1_codes: u8,
    d1_numbers: bool,
    d1_prompt: u8,
    thousandths_table: Option<SlotReadout>,
    yes_no_table: SlotReadout,
}

impl std::fmt::Debug for PromptBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptBuilder")
            .field("d1_codes", &self.d1_codes)
            .field("d1_numbers", &self.d1_numbers)
            .field("d1_prompt", &self.d1_prompt)
            .field("has_thousandths_table", &self.thousandths_table.is_some())
            .finish_non_exhaustive()
    }
}

impl PromptBuilder {
    /// Loads `tokenizer.json` (and optional `tokenizer_config.json`) from a checkpoint folder.
    ///
    /// # Errors
    /// Returns [`D1Error`] if the tokenizer cannot be loaded or lacks single-token `yes`/`no`.
    pub fn from_checkpoint_dir(dir: impl AsRef<Path>, config: &ModelConfig) -> Result<Self> {
        let dir = dir.as_ref();
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| D1Error::Tokenizer(format!("loading {}: {e}", tok_path.display())))?;
        let meta = TokenizerMeta::load_or_default(dir.join("tokenizer_config.json"))?;
        Self::from_parts(tokenizer, meta, config)
    }

    /// Constructs a [`PromptBuilder`] from an in-memory [`Tokenizer`] and [`ModelConfig`].
    ///
    /// # Errors
    /// Returns [`D1Error`] if the tokenizer lacks single-token `yes`/`no` forms.
    pub fn from_parts(
        tokenizer: Tokenizer,
        meta: TokenizerMeta,
        config: &ModelConfig,
    ) -> Result<Self> {
        let escape_regex = build_special_escape_regex(&tokenizer, &meta);
        let yes_ids = single_token_ids(&tokenizer, &YES_FORMS)?;
        let no_ids = single_token_ids(&tokenizer, &NO_FORMS)?;
        if yes_ids.is_empty() || no_ids.is_empty() {
            return Err(D1Error::Tokenizer(
                "tokenizer lacks single-token forms for yes/no".to_string(),
            ));
        }
        let yes_no_table =
            build_readout_from_groups(&[yes_ids, no_ids], Some(CalibrationKind::Noul));

        let thousandths_table = if config.d1_numbers {
            build_number_readout(&tokenizer, &THOUSANDTHS_GRID, 1000, None)?
        } else {
            None
        };

        Ok(Self {
            tokenizer,
            meta,
            escape_regex,
            d1_codes: config.d1_codes,
            d1_numbers: config.d1_numbers,
            d1_prompt: config.d1_prompt,
            thousandths_table,
            yes_no_table,
        })
    }

    /// Escapes any tokenizer special-token literal in caller text with `\u{200b}`.
    #[must_use]
    pub fn escape_plain(&self, text: &str) -> String {
        let Some(re) = &self.escape_regex else {
            return text.to_string();
        };
        let mut current = text.to_string();
        while re.is_match(&current) {
            current = re
                .replace_all(&current, |caps: &regex::Captures<'_>| {
                    let matched = &caps[0];
                    let mut chars = matched.chars();
                    let first = chars.next().unwrap_or_default();
                    let rest: String = chars.collect();
                    format!("{first}{ZERO_WIDTH_BREAK}{rest}")
                })
                .into_owned();
        }
        current
    }

    /// Encodes `text` without adding automatic BOS/EOS tokens.
    ///
    /// # Errors
    /// Returns [`D1Error::Tokenizer`] if tokenization fails.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Renders the shared prefix (`<|startoftext|><|im_start|>user\n` + expanded vision markup + optional text state).
    #[must_use]
    pub fn render_prefix(&self, num_image_tokens: usize, state: Option<&str>) -> String {
        let media = if num_image_tokens > 0 {
            let repeated = self.meta.image_token.repeat(num_image_tokens);
            format!(
                "{}{}{}",
                self.meta.vision_start_token, repeated, self.meta.vision_end_token
            )
        } else {
            String::new()
        };

        let body = match state {
            None => String::new(),
            Some(s) => {
                let escaped = self.escape_plain(&format!("{s}\n\n"));
                format!("{escaped}\nQUESTION:\n")
            }
        };

        format!("{}{IM_START}user\n{media}{body}", self.meta.bos_token)
    }

    /// Packs `questions` (up to `set_size` per pass) after the shared image/state prefix.
    ///
    /// # Errors
    /// Returns [`D1Error`] if any question cannot be verbalized or tokenized cleanly.
    pub fn pack_questions(
        &self,
        num_image_tokens: usize,
        state: Option<&str>,
        questions: &[Question],
        set_size: usize,
    ) -> Result<Vec<PackedGroup>> {
        if questions.is_empty() {
            return Err(D1Error::InvalidRequest(
                "questions map cannot be empty".to_string(),
            ));
        }
        if questions.iter().any(Question::requires_single_image) && num_image_tokens == 0 {
            return Err(D1Error::InvalidRequest(
                "box/boxes questions require an input image frame".to_string(),
            ));
        }

        let prefix = self.render_prefix(num_image_tokens, state);
        let header = if state.is_some() { "QUESTION:\n" } else { "" };

        let mut suffixes = Vec::with_capacity(questions.len());
        let mut slot_layouts = Vec::with_capacity(questions.len());
        let mut slot_readouts = Vec::with_capacity(questions.len());

        for q in questions {
            let (suffix, offsets) = self.render_suffix_layout(q)?;
            let readouts = self.question_slot_readouts(q)?;
            if offsets.len() != readouts.len() {
                return Err(D1Error::InvalidConfig(format!(
                    "slot offset count ({}) != readout count ({})",
                    offsets.len(),
                    readouts.len()
                )));
            }
            suffixes.push(suffix);
            slot_layouts.push(offsets);
            slot_readouts.push(readouts);
        }

        let step = set_size.max(1);
        let mut groups = Vec::new();

        for start in (0..questions.len()).step_by(step) {
            let end = (start + step).min(questions.len());
            let first_full = format!("{prefix}{}", suffixes[start]);
            let mut ids = self.encode_raw(&first_full)?;
            let first_suffix_ids = self.encode_raw(&suffixes[start])?;
            if ids.len() < first_suffix_ids.len() {
                return Err(D1Error::Tokenizer(
                    "tokenized prompt shorter than first question suffix".to_string(),
                ));
            }
            let state_len = ids.len() - first_suffix_ids.len();
            let mut question_groups = vec![-1_i32; state_len];
            question_groups.resize(ids.len(), 1);

            let mut question_slots = Vec::with_capacity(end - start);
            question_slots.push(resolve_slot_positions(
                ids.len() - state_len,
                ids.len() - 1,
                &slot_layouts[start],
                &slot_readouts[start],
            )?);

            for (group_idx, q_idx) in ((start + 1)..end).enumerate() {
                let group_num = (group_idx + 2) as i32;
                let extra_text = format!("{header}{}", suffixes[q_idx]);
                let own_ids = self.encode_raw(&extra_text)?;
                ids.extend_from_slice(&own_ids);
                question_groups.resize(ids.len(), group_num);
                question_slots.push(resolve_slot_positions(
                    own_ids.len(),
                    ids.len() - 1,
                    &slot_layouts[q_idx],
                    &slot_readouts[q_idx],
                )?);
            }

            groups.push(PackedGroup {
                input_ids: ids,
                state_len,
                question_groups,
                question_slots,
            });
        }

        Ok(groups)
    }

    fn render_suffix_layout(&self, q: &Question) -> Result<(String, Vec<isize>)> {
        let body = self.render_question_body(q)?;
        match q {
            Question::Box { .. } => {
                let mut markers = Vec::with_capacity(5);
                markers.push("present".to_string());
                markers.extend(EDGE_NAMES.iter().map(|s| (*s).to_string()));
                self.marked_suffix(&body, &markers, "")
            }
            Question::Boxes { .. } => {
                let blocks = q.box_blocks();
                let mut markers = Vec::with_capacity(blocks * 5);
                for k in 1..=blocks {
                    markers.push(format!("present {k}"));
                    for edge in EDGE_NAMES {
                        markers.push(format!("{edge} {k}"));
                    }
                }
                let after = if blocks < 16 || self.d1_prompt > 1 {
                    "\n"
                } else {
                    ""
                };
                self.marked_suffix(&body, &markers, after)
            }
            Question::MultipleChoice { criteria, .. } => {
                let markers: Vec<String> = criteria
                    .keys()
                    .map(|k| serde_json::to_string(k).unwrap_or_else(|_| format!("\"{k}\"")))
                    .collect();
                self.marked_suffix(&body, &markers, "")
            }
            Question::Choice { .. }
            | Question::Noul { .. }
            | Question::Score { .. }
            | Question::Estimate { .. } => {
                let escaped = self.escape_plain(&body);
                let text = format!("{escaped}{IM_END}\n{IM_START}assistant\n{LFM3_LEAD}");
                Ok((text, vec![0]))
            }
        }
    }

    fn marked_suffix(
        &self,
        body: &str,
        names: &[String],
        after: &str,
    ) -> Result<(String, Vec<isize>)> {
        let escaped_body = self.escape_plain(body);
        let mut text = format!("{escaped_body}{IM_END}\n{IM_START}assistant\n{LFM3_LEAD}");
        let mut colon_ends = Vec::with_capacity(names.len());

        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(&self.escape_plain(name));
            text.push(':');
            colon_ends.push(text.len());
        }

        if !after.is_empty() {
            let candidate = format!("{text}{after}");
            let cand_ids = self.encode_raw(&candidate)?;
            if let Some(&last_id) = cand_ids.last() {
                let decoded = self
                    .tokenizer
                    .decode(&[last_id], false)
                    .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
                if decoded.trim_end().ends_with(':') {
                    text = candidate;
                }
            }
        }

        let enc = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
        let ids = enc.get_ids();
        let offsets = enc.get_offsets();
        let total_tokens = ids.len() as isize;

        let mut slot_offsets = Vec::with_capacity(colon_ends.len());
        for &byte_end in &colon_ends {
            let colon_byte = byte_end - 1;
            let matches: Vec<(usize, usize)> = offsets
                .iter()
                .enumerate()
                .filter_map(|(tok_idx, &(start, stop))| {
                    (start <= colon_byte && colon_byte < stop).then_some((tok_idx, stop))
                })
                .collect();
            if matches.len() != 1 {
                return Err(D1Error::Tokenizer(format!(
                    "expected unique colon token at byte {colon_byte}, found {}",
                    matches.len()
                )));
            }
            let (tok_idx, stop) = matches[0];
            if stop > byte_end && !text[byte_end..stop].trim().is_empty() {
                return Err(D1Error::Tokenizer(
                    "slot colon token merged with following marker text".to_string(),
                ));
            }
            slot_offsets.push((tok_idx as isize) - total_tokens + 1);
        }

        Ok((text, slot_offsets))
    }

    fn render_question_body(&self, q: &Question) -> Result<String> {
        match q {
            Question::Box { instructions } => {
                let (legend, unit_word) = self.edge_legend_and_word()?;
                Ok(format!(
                    "{instructions}\n\nPresence is yes or no. {legend} Reply with yes or no at present, then {unit_word} at top, bottom, left and right."
                ))
            }
            Question::Boxes { instructions, .. } => {
                let (legend, _) = self.edge_legend_and_word()?;
                let most = if self.d1_prompt == 1 {
                    "16 at most, "
                } else {
                    ""
                };
                Ok(format!(
                    "{instructions}\n\nEvery one of them, {most}in reading order (left edge, then top). Present k is yes when there are at least k. {legend} Reply with yes or no at each present, then its top, bottom, left and right."
                ))
            }
            Question::Estimate {
                instructions, unit, ..
            } => {
                let grid = q.estimate_grid()?;
                let unit_suffix = unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
                if self.d1_numbers
                    && build_number_readout(&self.tokenizer, &grid, 1, None)?.is_some()
                {
                    Ok(format!(
                        "{instructions}\n\nEstimate from {} to {}{unit_suffix}, in steps of {}. Reply with the number only.",
                        format_general(grid.low()),
                        format_general(grid.high()),
                        format_general(grid.step()),
                    ))
                } else {
                    let labels: Vec<String> =
                        grid.values().into_iter().map(format_general).collect();
                    let codes = if grid.size() <= 10 {
                        (0..grid.size()).map(|i| i.to_string()).collect::<Vec<_>>()
                    } else {
                        let keys: Vec<String> = (0..labels.len()).map(|i| i.to_string()).collect();
                        self.aliases(&keys)?
                            .into_iter()
                            .map(|(code, _)| code)
                            .collect()
                    };
                    let first = codes.first().cloned().unwrap_or_default();
                    let last = codes.last().cloned().unwrap_or_default();
                    Ok(format!(
                        "{instructions}\n\nEstimate from {} to {}{unit_suffix} on {} equally spaced values, step {}: {first} is {}, {last} is {}. Reply with the code only.",
                        format_general(grid.low()),
                        format_general(grid.high()),
                        grid.size(),
                        format_general(grid.step()),
                        format_general(grid.low()),
                        format_general(grid.high()),
                    ))
                }
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                let labels: Vec<String> = criteria.keys().cloned().collect();
                let coded = self.aliases(&labels)?;
                let mut lines = Vec::with_capacity(labels.len());
                for ((code, _), label) in coded.iter().zip(labels.iter()) {
                    let desc = criteria
                        .get(label)
                        .and_then(|d| d.as_deref())
                        .map_or_else(|| label.replace('_', " "), ToString::to_string);
                    lines.push(format!("{code} {desc}"));
                }
                Ok(format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                ))
            }
            Question::MultipleChoice {
                instructions,
                criteria,
            } => {
                let mut lines = Vec::with_capacity(criteria.len());
                for (name, desc) in criteria {
                    let quoted =
                        serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""));
                    let shown = desc.as_deref().unwrap_or(name.as_str());
                    lines.push(format!("{quoted}: {shown}"));
                }
                Ok(format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with yes or no for each option.",
                    lines.join("\n")
                ))
            }
            Question::Noul {
                instructions,
                criteria,
            } => {
                let extra = match criteria {
                    Some(c) => format!("\nYes: {}\nNo: {}", c.yes, c.no),
                    None => String::new(),
                };
                Ok(format!(
                    "{instructions}{extra}\n\nReply with yes or no only."
                ))
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                let legend: Vec<String> = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, text)| format!("{i} {text}"))
                    .collect();
                Ok(format!(
                    "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                    legend.join("\n"),
                    criteria.len() - 1
                ))
            }
        }
    }

    fn edge_legend_and_word(&self) -> Result<(String, &'static str)> {
        if self.thousandths_table.is_some() {
            return Ok((
                "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge.".to_string(),
                "one number",
            ));
        }
        let keys: Vec<String> = (0..GRID_128.size()).map(|i| i.to_string()).collect();
        let coded = self.aliases(&keys)?;
        let first = coded.first().map(|c| c.0.as_str()).unwrap_or("A");
        let last = coded.last().map(|c| c.0.as_str()).unwrap_or("Z");
        Ok((
            format!(
                "Each edge is a code on a {}-value scale: {first} is 0 (top or left), {last} is 1 (bottom or right). Values are equally spaced.",
                GRID_128.size()
            ),
            "one code",
        ))
    }

    fn question_slot_readouts(&self, q: &Question) -> Result<Vec<SlotReadout>> {
        match q {
            Question::Choice { criteria, .. } => {
                let labels: Vec<String> = criteria.keys().cloned().collect();
                let coded = self.aliases(&labels)?;
                let mut groups = Vec::with_capacity(coded.len());
                for (code, tid) in coded {
                    let spaced = format!(" {code}");
                    let extra = single_token_ids(&self.tokenizer, &[spaced.as_str()])?;
                    let mut group = vec![tid];
                    for id in extra {
                        if id != tid {
                            group.push(id);
                        }
                    }
                    groups.push(group);
                }
                Ok(vec![build_readout_from_groups(
                    &groups,
                    Some(CalibrationKind::Choice(criteria.len())),
                )])
            }
            Question::MultipleChoice { criteria, .. } => {
                let mut untempered = self.yes_no_table.clone();
                untempered.calibration_kind = None;
                Ok(vec![untempered; criteria.len()])
            }
            Question::Noul { .. } => Ok(vec![self.yes_no_table.clone()]),
            Question::Score { criteria, .. } => {
                let mut groups = Vec::with_capacity(criteria.len());
                for i in 0..criteria.len() {
                    let s = i.to_string();
                    let g = single_token_ids(&self.tokenizer, &[s.as_str()])?;
                    if g.is_empty() {
                        return Err(D1Error::Tokenizer(format!(
                            "missing single-token digit for score level {i}"
                        )));
                    }
                    groups.push(g);
                }
                Ok(vec![build_readout_from_groups(
                    &groups,
                    Some(CalibrationKind::Score(criteria.len())),
                )])
            }
            Question::Estimate { .. } => {
                let grid = q.estimate_grid()?;
                if self.d1_numbers {
                    if let Some(readout) = build_number_readout(&self.tokenizer, &grid, 1, None)? {
                        return Ok(vec![readout]);
                    }
                }
                Ok(vec![self.grid_code_readout(&grid, None)?])
            }
            Question::Box { .. } => self.single_box_readouts(),
            Question::Boxes { .. } => {
                let one = self.single_box_readouts()?;
                let blocks = q.box_blocks();
                let mut out = Vec::with_capacity(blocks * 5);
                for _ in 0..blocks {
                    out.extend(one.iter().cloned());
                }
                Ok(out)
            }
        }
    }

    fn single_box_readouts(&self) -> Result<Vec<SlotReadout>> {
        let mut presence = self.yes_no_table.clone();
        presence.calibration_kind = None;
        let edge_readout = match &self.thousandths_table {
            Some(tbl) => tbl.clone(),
            None => self.grid_code_readout(&GRID_128, None)?,
        };
        Ok(vec![
            presence,
            edge_readout.clone(),
            edge_readout.clone(),
            edge_readout.clone(),
            edge_readout,
        ])
    }

    fn grid_code_readout(
        &self,
        grid: &Grid,
        cal_kind: Option<CalibrationKind>,
    ) -> Result<SlotReadout> {
        if grid.size() <= 10 {
            let mut groups = Vec::with_capacity(grid.size());
            for i in 0..grid.size() {
                let s = i.to_string();
                let g = single_token_ids(&self.tokenizer, &[s.as_str()])?;
                if g.is_empty() {
                    return Err(D1Error::Tokenizer(format!("missing digit token {i}")));
                }
                groups.push(g);
            }
            return Ok(build_readout_from_groups(&groups, cal_kind));
        }
        let keys: Vec<String> = (0..grid.size()).map(|i| i.to_string()).collect();
        let coded = self.aliases(&keys)?;
        let mut groups = Vec::with_capacity(coded.len());
        for (code, tid) in coded {
            let spaced = format!(" {code}");
            let extra = single_token_ids(&self.tokenizer, &[spaced.as_str()])?;
            let mut g = vec![tid];
            for id in extra {
                if id != tid {
                    g.push(id);
                }
            }
            groups.push(g);
        }
        Ok(build_readout_from_groups(&groups, cal_kind))
    }

    fn aliases(&self, labels: &[String]) -> Result<Vec<(String, u32)>> {
        let codes = option_codes(labels, self.d1_codes);
        let mut used = HashSet::with_capacity(codes.len());
        let mut out = Vec::with_capacity(codes.len());

        let mut try_take = |raw: &str| -> Result<bool> {
            let enc = self
                .tokenizer
                .encode(raw, false)
                .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
            let ids = enc.get_ids();
            if ids.len() != 1 || used.contains(&ids[0]) {
                return Ok(false);
            }
            used.insert(ids[0]);
            out.push((raw.to_string(), ids[0]));
            Ok(true)
        };

        for code in &codes {
            if try_take(code)? {
                continue;
            }
            let mut found = false;
            for fallback in fallback_pool() {
                if try_take(fallback)? {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(D1Error::Tokenizer(format!(
                    "exhausted single-token alias pool for {} options",
                    codes.len()
                )));
            }
        }
        Ok(out)
    }
}

/// Converts raw candidate logits for a [`SlotReadout`] into calibrated option probabilities.
#[must_use]
pub fn slot_probabilities(
    readout: &SlotReadout,
    candidate_logits: &[f32],
    calibration: Option<&Calibration>,
) -> Vec<f64> {
    let mut scores: Vec<f64> = readout
        .option_forms
        .iter()
        .map(|forms| {
            forms
                .iter()
                .map(|&idx| f64::from(candidate_logits[idx]))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    if let (Some(cal), Some(kind)) = (calibration, readout.calibration_kind) {
        cal.scale_scores(kind, &mut scores);
    }

    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let total: f64 = exps.iter().sum();
    if total <= 0.0 {
        vec![0.0; scores.len()]
    } else {
        exps.into_iter().map(|e| e / total).collect()
    }
}

fn resolve_slot_positions(
    segment_len: usize,
    end_pos: usize,
    offsets: &[isize],
    readouts: &[SlotReadout],
) -> Result<Vec<(usize, SlotReadout)>> {
    if offsets.is_empty() || *offsets.last().unwrap_or(&-1) != 0 {
        return Err(D1Error::Tokenizer(
            "question slot offsets must end at 0".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(offsets.len());
    for (&off, readout) in offsets.iter().zip(readouts.iter()) {
        if off > 0 || (-off as usize) >= segment_len {
            return Err(D1Error::Tokenizer(format!(
                "slot offset {off} outside question segment of length {segment_len}"
            )));
        }
        let abs_pos = ((end_pos as isize) + off) as usize;
        out.push((abs_pos, readout.clone()));
    }
    Ok(out)
}

fn option_codes(labels: &[String], scheme: u8) -> Vec<String> {
    let trimmed: Vec<&str> = labels.iter().map(|s| s.trim()).collect();
    if !trimmed.is_empty()
        && trimmed
            .iter()
            .all(|k| k.chars().count() == 1 && k.chars().all(char::is_alphabetic))
    {
        return trimmed.into_iter().map(ToString::to_string).collect();
    }
    if trimmed.len() <= 26 {
        return (0..trimmed.len())
            .map(|i| char::from(b'A' + (i as u8)).to_string())
            .collect();
    }
    if scheme == 2 {
        let mut codes: Vec<String> = (b'A'..=b'Z').map(|c| char::from(c).to_string()).collect();
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                if codes.len() >= trimmed.len() {
                    break;
                }
                codes.push(format!("{}{}", char::from(a), char::from(b)));
            }
            if codes.len() >= trimmed.len() {
                break;
            }
        }
        codes.truncate(trimmed.len());
        return codes;
    }
    (0..trimmed.len()).map(|i| format!("{i:02}")).collect()
}

fn fallback_pool() -> &'static [String] {
    static POOL: OnceLock<Vec<String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let mut p = Vec::with_capacity(1028);
        p.extend((b'A'..=b'Z').map(|c| char::from(c).to_string()));
        p.extend((0..100).map(|i| format!("{i:02}")));
        p.extend((b'a'..=b'z').map(|c| char::from(c).to_string()));
        p.extend((0..200).map(|i| format!("#{i}")));
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                p.push(format!("{}{}", char::from(a), char::from(b)));
            }
        }
        p
    })
}

fn single_token_ids(tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for &t in texts {
        let enc = tokenizer
            .encode(t, false)
            .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
        let ids = enc.get_ids();
        if ids.len() == 1 && seen.insert(ids[0]) {
            out.push(ids[0]);
        }
    }
    Ok(out)
}

fn build_number_readout(
    tokenizer: &Tokenizer,
    grid: &Grid,
    scale: i64,
    cal_kind: Option<CalibrationKind>,
) -> Result<Option<SlotReadout>> {
    let mut groups = Vec::with_capacity(grid.size());
    for val in grid.values() {
        let scaled = val * (scale as f64);
        let rounded = scaled.round();
        if (scaled - rounded).abs() > 1e-6 {
            return Ok(None);
        }
        let n = rounded as i64;
        let plain = n.to_string();
        let spaced = format!(" {n}");
        let plain_enc = tokenizer
            .encode(plain.as_str(), false)
            .map_err(|e| D1Error::Tokenizer(e.to_string()))?;
        if plain_enc.get_ids().len() != 1 {
            return Ok(None);
        }
        let group = single_token_ids(tokenizer, &[plain.as_str(), spaced.as_str()])?;
        if group.is_empty() {
            return Ok(None);
        }
        groups.push(group);
    }
    Ok(Some(build_readout_from_groups(&groups, cal_kind)))
}

fn build_readout_from_groups(
    groups: &[Vec<u32>],
    calibration_kind: Option<CalibrationKind>,
) -> SlotReadout {
    let mut unique_tokens = Vec::new();
    let mut pos_map: HashMap<u32, usize> = HashMap::new();
    for g in groups {
        for &tid in g {
            pos_map.entry(tid).or_insert_with(|| {
                let idx = unique_tokens.len();
                unique_tokens.push(tid);
                idx
            });
        }
    }
    let option_forms = groups
        .iter()
        .map(|g| g.iter().map(|tid| pos_map[tid]).collect())
        .collect();
    SlotReadout {
        unique_tokens,
        option_forms,
        calibration_kind,
    }
}

fn build_special_escape_regex(tokenizer: &Tokenizer, meta: &TokenizerMeta) -> Option<Regex> {
    let mut specials: HashSet<String> = tokenizer
        .get_added_tokens_decoder()
        .into_values()
        .filter(|tok| tok.special)
        .map(|tok| tok.content)
        .collect();
    for extra in [
        IM_START,
        IM_END,
        meta.bos_token.as_str(),
        meta.image_token.as_str(),
        meta.vision_start_token.as_str(),
        meta.vision_end_token.as_str(),
    ] {
        if !extra.is_empty() {
            specials.insert(extra.to_string());
        }
    }
    let mut list: Vec<String> = specials
        .into_iter()
        .filter(|s| s.chars().count() > 1 && !s.contains(ZERO_WIDTH_BREAK))
        .collect();
    if list.is_empty() {
        return None;
    }
    list.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let pattern = list
        .into_iter()
        .map(|s| regex::escape(&s))
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&pattern).ok()
}

fn format_general(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e12 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v}")
    }
}
