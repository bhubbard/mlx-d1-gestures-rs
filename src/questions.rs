//! Typed System-One questions, answers, and the default webcam gesture/expression schema.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::box_decode::{
    decode_many_boxes, decode_single_box, DetectedBoxItem, NormalizedBox, DEFAULT_BOX_INSTANCES,
    MAX_BOX_INSTANCES,
};
use crate::calibration::Calibration;
use crate::config::BoxEdgeRead;
use crate::error::{D1Error, Result};
use crate::grid::Grid;

/// Aggregation rule turning an ordinal `score` distribution into a scalar level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreReadout {
    /// Cumulative median level (optimal for ordinal absolute error; d1 default).
    #[default]
    Median,
    /// Expected level `sum(i * p_i)`.
    Mean,
    /// Most likely level `argmax_i p_i`.
    Argmax,
}

/// Optional yes/no criteria labels for a `noul` binary question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryCriteria {
    /// Meaning of `"yes"`.
    #[serde(rename = "true")]
    pub yes: String,
    /// Meaning of `"no"`.
    #[serde(rename = "false")]
    pub no: String,
}

/// Validated System-One question over an input image or text state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawQuestion", into = "RawQuestion")]
pub enum Question {
    /// Single-pick categorical choice (`{"type": "choice", "criteria": {...}}`).
    Choice {
        /// Natural-language question text.
        instructions: String,
        /// Ordered map from option key to human-readable description.
        criteria: IndexMap<String, Option<String>>,
    },
    /// Independent multi-label yes/no over each option (`"multiple": true`).
    MultipleChoice {
        /// Natural-language question text.
        instructions: String,
        /// Ordered map from option key to human-readable description.
        criteria: IndexMap<String, Option<String>>,
    },
    /// Binary yes/no probability (`{"type": "noul"}`).
    Noul {
        /// Natural-language question text.
        instructions: String,
        /// Optional description of yes vs no.
        criteria: Option<BinaryCriteria>,
    },
    /// Ordinal 2..=10 level rubric (`{"type": "score", "criteria": [...]}`).
    Score {
        /// Natural-language question text.
        instructions: String,
        /// Ordered level descriptions from lowest (`0`) to highest.
        criteria: Vec<String>,
    },
    /// Numeric range estimation (`{"type": "number", "range": [low, high]}`).
    Estimate {
        /// Natural-language question text.
        instructions: String,
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
        /// Optional step size dividing `high - low`.
        step: Option<f64>,
        /// Optional physical unit (`"ms"`, `"kg"`, etc.).
        unit: Option<String>,
    },
    /// Single normalized bounding box (`{"type": "box"}`).
    Box {
        /// Natural-language localization prompt.
        instructions: String,
    },
    /// Multi-instance bounding boxes in reading order (`{"type": "boxes"}`).
    Boxes {
        /// Natural-language localization prompt.
        instructions: String,
        /// Optional cap on instance blocks (`1..=16` on prompt v1, `1..=64` on prompt v2).
        max: Option<usize>,
    },
}

impl Question {
    /// Natural-language question instructions.
    #[must_use]
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::MultipleChoice { instructions, .. }
            | Self::Noul { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Estimate { instructions, .. }
            | Self::Box { instructions }
            | Self::Boxes { instructions, .. } => instructions,
        }
    }

    /// Whether this question uses untempered raw slot probabilities calibrated via knot tables.
    #[must_use]
    pub const fn is_typed(&self) -> bool {
        matches!(
            self,
            Self::MultipleChoice { .. }
                | Self::Estimate { .. }
                | Self::Box { .. }
                | Self::Boxes { .. }
        )
    }

    /// Whether this question requires a single input image (`box` or `boxes`).
    #[must_use]
    pub const fn requires_single_image(&self) -> bool {
        matches!(self, Self::Box { .. } | Self::Boxes { .. })
    }

    /// Grid constructed for an [`Question::Estimate`] variant.
    ///
    /// # Errors
    /// Returns [`D1Error`] if called on a non-estimate question or invalid bounds.
    pub fn estimate_grid(&self) -> Result<Grid> {
        match self {
            Self::Estimate {
                low, high, step, ..
            } => {
                let size = match step {
                    None => 128,
                    Some(s) => (((high - low) / s).round() as usize + 1).min(128),
                };
                Grid::try_new(*low, *high, size)
            }
            _ => Err(D1Error::InvalidRequest(
                "estimate_grid called on non-estimate question".to_string(),
            )),
        }
    }

    /// Number of 5-slot blocks evaluated for a [`Question::Boxes`] query.
    #[must_use]
    pub fn box_blocks(&self) -> usize {
        match self {
            Self::Boxes { max: Some(m), .. } => *m,
            Self::Boxes { max: None, .. } => DEFAULT_BOX_INSTANCES,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawQuestion {
    r#type: String,
    instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    criteria: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    multiple: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<usize>,
}

impl TryFrom<RawQuestion> for Question {
    type Error = D1Error;

    fn try_from(raw: RawQuestion) -> Result<Self> {
        if raw.instructions.trim().is_empty() {
            return Err(D1Error::InvalidRequest(
                "question `instructions` must be a non-empty string".to_string(),
            ));
        }
        match raw.r#type.as_str() {
            "choice" => {
                let Some(val) = raw.criteria else {
                    return Err(D1Error::InvalidRequest(
                        "choice question requires `criteria` object".to_string(),
                    ));
                };
                let criteria: IndexMap<String, Option<String>> = serde_json::from_value(val)
                    .map_err(|e| {
                        D1Error::InvalidRequest(format!("invalid choice criteria map: {e}"))
                    })?;
                if criteria.is_empty() {
                    return Err(D1Error::InvalidRequest(
                        "choice `criteria` must contain at least one option".to_string(),
                    ));
                }
                if raw.multiple.unwrap_or(false) {
                    Ok(Self::MultipleChoice {
                        instructions: raw.instructions,
                        criteria,
                    })
                } else {
                    Ok(Self::Choice {
                        instructions: raw.instructions,
                        criteria,
                    })
                }
            }
            "noul" | "bool" => {
                let criteria = match raw.criteria {
                    Some(v) if !v.is_null() => Some(serde_json::from_value(v).map_err(|e| {
                        D1Error::InvalidRequest(format!("invalid noul criteria: {e}"))
                    })?),
                    _ => None,
                };
                Ok(Self::Noul {
                    instructions: raw.instructions,
                    criteria,
                })
            }
            "score" => {
                let Some(val) = raw.criteria else {
                    return Err(D1Error::InvalidRequest(
                        "score question requires `criteria` array of 2..=10 levels".to_string(),
                    ));
                };
                let criteria: Vec<String> = serde_json::from_value(val).map_err(|e| {
                    D1Error::InvalidRequest(format!("invalid score criteria list: {e}"))
                })?;
                if !(2..=10).contains(&criteria.len()) {
                    return Err(D1Error::InvalidRequest(format!(
                        "score `criteria` requires 2..=10 levels, got {}",
                        criteria.len()
                    )));
                }
                Ok(Self::Score {
                    instructions: raw.instructions,
                    criteria,
                })
            }
            "number" => {
                let Some([low, high]) = raw.range else {
                    return Err(D1Error::InvalidRequest(
                        "number estimate question requires `\"range\": [low, high]`".to_string(),
                    ));
                };
                if !low.is_finite() || !high.is_finite() || low >= high {
                    return Err(D1Error::InvalidRequest(format!(
                        "estimate range requires finite low < high, got [{low}, {high}]"
                    )));
                }
                if let Some(step) = raw.step {
                    let ratio = (high - low) / step;
                    if step <= 0.0 || (ratio - ratio.round()).abs() > 1e-6 {
                        return Err(D1Error::InvalidRequest(format!(
                            "estimate step {step} must be > 0 and divide [{low}, {high}]"
                        )));
                    }
                }
                Ok(Self::Estimate {
                    instructions: raw.instructions,
                    low,
                    high,
                    step: raw.step,
                    unit: raw.unit,
                })
            }
            "box" => Ok(Self::Box {
                instructions: raw.instructions,
            }),
            "boxes" => {
                if let Some(m) = raw.max {
                    if m == 0 || m > MAX_BOX_INSTANCES {
                        return Err(D1Error::InvalidRequest(format!(
                            "boxes `max` must be in 1..={MAX_BOX_INSTANCES}, got {m}"
                        )));
                    }
                }
                Ok(Self::Boxes {
                    instructions: raw.instructions,
                    max: raw.max,
                })
            }
            other => Err(D1Error::InvalidRequest(format!(
                "unsupported question type {other:?}"
            ))),
        }
    }
}

impl From<Question> for RawQuestion {
    fn from(q: Question) -> Self {
        match q {
            Question::Choice {
                instructions,
                criteria,
            } => Self {
                r#type: "choice".to_string(),
                instructions,
                criteria: serde_json::to_value(criteria).ok(),
                multiple: None,
                range: None,
                step: None,
                unit: None,
                max: None,
            },
            Question::MultipleChoice {
                instructions,
                criteria,
            } => Self {
                r#type: "choice".to_string(),
                instructions,
                criteria: serde_json::to_value(criteria).ok(),
                multiple: Some(true),
                range: None,
                step: None,
                unit: None,
                max: None,
            },
            Question::Noul {
                instructions,
                criteria,
            } => Self {
                r#type: "noul".to_string(),
                instructions,
                criteria: criteria.and_then(|c| serde_json::to_value(c).ok()),
                multiple: None,
                range: None,
                step: None,
                unit: None,
                max: None,
            },
            Question::Score {
                instructions,
                criteria,
            } => Self {
                r#type: "score".to_string(),
                instructions,
                criteria: serde_json::to_value(criteria).ok(),
                multiple: None,
                range: None,
                step: None,
                unit: None,
                max: None,
            },
            Question::Estimate {
                instructions,
                low,
                high,
                step,
                unit,
            } => Self {
                r#type: "number".to_string(),
                instructions,
                criteria: None,
                multiple: None,
                range: Some([low, high]),
                step,
                unit,
                max: None,
            },
            Question::Box { instructions } => Self {
                r#type: "box".to_string(),
                instructions,
                criteria: None,
                multiple: None,
                range: None,
                step: None,
                unit: None,
                max: None,
            },
            Question::Boxes { instructions, max } => Self {
                r#type: "boxes".to_string(),
                instructions,
                criteria: None,
                multiple: None,
                range: None,
                step: None,
                unit: None,
                max,
            },
        }
    }
}

/// Typed System-One answer serialized identically to upstream `decide.py` / `mlx-vlm`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawAnswer", into = "RawAnswer")]
pub enum Answer {
    /// Single-option categorical answer with full probability distribution.
    Choice {
        /// Winning option key.
        choice: String,
        /// Probability of the winning option.
        confidence: f64,
        /// Calibrated probability distribution across all options.
        probabilities: IndexMap<String, f64>,
    },
    /// Multi-label choice answer (`"multiple": true`).
    MultipleChoice {
        /// Option keys with calibrated `P(yes) >= 0.5`.
        choices: Vec<String>,
        /// Independent calibrated `P(yes)` per option.
        probabilities: IndexMap<String, f64>,
    },
    /// Binary `P(yes)` answer.
    Noul {
        /// Calibrated probability of `"yes"`.
        noul: f64,
    },
    /// Ordinal score answer.
    Score {
        /// Selected or expected score level.
        score: f64,
        /// Highest single-level probability.
        confidence: f64,
        /// Probability per level `"0"`..`"k-1"`.
        probabilities: IndexMap<String, f64>,
        /// Level descriptions keyed by `"0"`..`"k-1"`.
        legend: IndexMap<String, String>,
    },
    /// Numeric grid estimate answer.
    Number {
        /// Median estimate.
        number: f64,
        /// Central 80% `[p10, p90]` interval.
        interval: [f64; 2],
        /// Optional unit string.
        unit: Option<String>,
    },
    /// Single bounding-box answer (`box` is `null` when `found < 0.5`).
    Box {
        /// Normalized `[0, 1]` coordinates when detected.
        r#box: Option<NormalizedBox>,
        /// Calibrated presence probability.
        found: f64,
        /// Calibrated IoU>=0.5 confidence (`0.0` when absent).
        confidence: f64,
    },
    /// Multi-instance bounding-box list in reading order.
    Boxes {
        /// Detected boxes with `found >= 0.5`.
        boxes: Vec<DetectedBoxItem>,
        /// Number of detected instances.
        count: usize,
        /// `true` when every evaluated block was present.
        at_least: bool,
        /// Monotone non-increasing `P(count >= k)` curve.
        presence: Vec<f64>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RawAnswer {
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choices: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    probabilities: Option<IndexMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    noul: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    legend: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    number: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    r#box: Option<Option<NormalizedBox>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    found: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boxes: Option<Vec<DetectedBoxItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    at_least: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence: Option<Vec<f64>>,
}

impl TryFrom<RawAnswer> for Answer {
    type Error = D1Error;

    fn try_from(raw: RawAnswer) -> Result<Self> {
        match raw.r#type.as_str() {
            "choice" => {
                let probabilities = raw.probabilities.unwrap_or_default();
                if let Some(choices) = raw.choices {
                    Ok(Self::MultipleChoice {
                        choices,
                        probabilities,
                    })
                } else {
                    let choice = raw.choice.ok_or_else(|| {
                        D1Error::InvalidRequest("missing `choice` in choice answer".to_string())
                    })?;
                    Ok(Self::Choice {
                        choice,
                        confidence: raw.confidence.unwrap_or(0.0),
                        probabilities,
                    })
                }
            }
            "noul" => Ok(Self::Noul {
                noul: raw.noul.unwrap_or(0.0),
            }),
            "score" => Ok(Self::Score {
                score: raw.score.unwrap_or(0.0),
                confidence: raw.confidence.unwrap_or(0.0),
                probabilities: raw.probabilities.unwrap_or_default(),
                legend: raw.legend.unwrap_or_default(),
            }),
            "number" => Ok(Self::Number {
                number: raw.number.unwrap_or(0.0),
                interval: raw.interval.unwrap_or([0.0, 0.0]),
                unit: raw.unit,
            }),
            "box" => Ok(Self::Box {
                r#box: raw.r#box.flatten(),
                found: raw.found.unwrap_or(0.0),
                confidence: raw.confidence.unwrap_or(0.0),
            }),
            "boxes" => Ok(Self::Boxes {
                boxes: raw.boxes.unwrap_or_default(),
                count: raw.count.unwrap_or(0),
                at_least: raw.at_least.unwrap_or(false),
                presence: raw.presence.unwrap_or_default(),
            }),
            other => Err(D1Error::InvalidRequest(format!(
                "unknown answer type {other:?}"
            ))),
        }
    }
}

impl From<Answer> for RawAnswer {
    fn from(ans: Answer) -> Self {
        match ans {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => Self {
                r#type: "choice".to_string(),
                choice: Some(choice),
                confidence: Some(confidence),
                probabilities: Some(probabilities),
                ..Self::default()
            },
            Answer::MultipleChoice {
                choices,
                probabilities,
            } => Self {
                r#type: "choice".to_string(),
                choices: Some(choices),
                probabilities: Some(probabilities),
                ..Self::default()
            },
            Answer::Noul { noul } => Self {
                r#type: "noul".to_string(),
                noul: Some(noul),
                ..Self::default()
            },
            Answer::Score {
                score,
                confidence,
                probabilities,
                legend,
            } => Self {
                r#type: "score".to_string(),
                score: Some(score),
                confidence: Some(confidence),
                probabilities: Some(probabilities),
                legend: Some(legend),
                ..Self::default()
            },
            Answer::Number {
                number,
                interval,
                unit,
            } => Self {
                r#type: "number".to_string(),
                number: Some(number),
                interval: Some(interval),
                unit,
                ..Self::default()
            },
            Answer::Box {
                r#box,
                found,
                confidence,
            } => Self {
                r#type: "box".to_string(),
                r#box: Some(r#box),
                found: Some(found),
                confidence: Some(confidence),
                ..Self::default()
            },
            Answer::Boxes {
                boxes,
                count,
                at_least,
                presence,
            } => Self {
                r#type: "boxes".to_string(),
                boxes: Some(boxes),
                count: Some(count),
                at_least: Some(at_least),
                presence: Some(presence),
                ..Self::default()
            },
        }
    }
}

/// Converts raw slot probability vectors into a calibrated [`Answer`].
///
/// # Errors
/// Returns [`D1Error`] if `slot_probs` does not match the question's slot count.
pub fn build_answer(
    question: &Question,
    slot_probs: &[Vec<f64>],
    score_readout: ScoreReadout,
    calibration: Option<&Calibration>,
    edge_mode: BoxEdgeRead,
) -> Result<Answer> {
    match question {
        Question::Choice { criteria, .. } => {
            let probs = slot_probs.first().ok_or_else(|| {
                D1Error::InvalidRequest("missing slot probabilities for choice".to_string())
            })?;
            let (best_idx, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or_else(|| D1Error::InvalidRequest("empty choice probabilities".to_string()))?;
            let names: Vec<&String> = criteria.keys().collect();
            let mut map = IndexMap::with_capacity(names.len());
            for (k, &p) in names.iter().zip(probs.iter()) {
                map.insert((*k).clone(), p);
            }
            Ok(Answer::Choice {
                choice: names[best_idx].clone(),
                confidence,
                probabilities: map,
            })
        }
        Question::MultipleChoice { criteria, .. } => {
            let mut choices = Vec::new();
            let mut probabilities = IndexMap::with_capacity(criteria.len());
            for (name, slot) in criteria.keys().zip(slot_probs.iter()) {
                let raw_yes = slot.first().copied().unwrap_or(0.0);
                let p = match calibration {
                    Some(cal) => cal.through("multiple", raw_yes),
                    None => raw_yes,
                };
                if p >= 0.5 {
                    choices.push(name.clone());
                }
                probabilities.insert(name.clone(), p);
            }
            Ok(Answer::MultipleChoice {
                choices,
                probabilities,
            })
        }
        Question::Noul { .. } => {
            let yes = slot_probs
                .first()
                .and_then(|p| p.first().copied())
                .unwrap_or(0.0);
            Ok(Answer::Noul { noul: yes })
        }
        Question::Score { criteria, .. } => {
            let probs = slot_probs.first().ok_or_else(|| {
                D1Error::InvalidRequest("missing slot probabilities for score".to_string())
            })?;
            let total: f64 = probs.iter().copied().sum();
            let norm = if total > 0.0 { total } else { 1.0 };
            let (_, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            let score = match score_readout {
                ScoreReadout::Mean => probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as f64) * (p / norm))
                    .sum(),
                ScoreReadout::Median => {
                    let mut mass = 0.0;
                    let mut chosen = probs.len().saturating_sub(1);
                    for (i, &p) in probs.iter().enumerate() {
                        mass += p / norm;
                        if mass >= 0.5 {
                            chosen = i;
                            break;
                        }
                    }
                    chosen as f64
                }
                ScoreReadout::Argmax => probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i) as f64,
            };
            let mut prob_map = IndexMap::with_capacity(criteria.len());
            let mut legend = IndexMap::with_capacity(criteria.len());
            for (i, text) in criteria.iter().enumerate() {
                prob_map.insert(i.to_string(), probs.get(i).copied().unwrap_or(0.0));
                legend.insert(i.to_string(), text.clone());
            }
            Ok(Answer::Score {
                score,
                confidence,
                probabilities: prob_map,
                legend,
            })
        }
        Question::Estimate { unit, .. } => {
            let probs = slot_probs.first().ok_or_else(|| {
                D1Error::InvalidRequest("missing slot probabilities for estimate".to_string())
            })?;
            let grid = question.estimate_grid()?;
            let stats = match calibration {
                Some(cal) => grid.read(probs, Some(|share| cal.level("estimate", share))),
                None => grid.read(probs, None::<fn(f64) -> f64>),
            };
            Ok(Answer::Number {
                number: stats.median,
                interval: [stats.p10, stats.p90],
                unit: unit.clone(),
            })
        }
        Question::Box { .. } => {
            let (rect, found, confidence) = decode_single_box(slot_probs, calibration, edge_mode);
            Ok(Answer::Box {
                r#box: rect,
                found,
                confidence,
            })
        }
        Question::Boxes { .. } => {
            let (boxes, at_least, presence) = decode_many_boxes(slot_probs, calibration, edge_mode);
            let count = boxes.len();
            Ok(Answer::Boxes {
                boxes,
                count,
                at_least,
                presence,
            })
        }
    }
}

/// Exact four-question webcam suite from upstream `ivanfioravanti/mlx-d1-gestures` (`decide.py`).
#[must_use]
pub fn default_gesture_questions() -> IndexMap<String, Question> {
    let mut hand_criteria = IndexMap::new();
    for (k, desc) in [
        ("thumbs_up", "A fist with the thumb pointing up"),
        ("thumbs_down", "A fist with the thumb pointing down"),
        (
            "ok",
            "Thumb and index finger touching in a circle, other fingers up",
        ),
        (
            "peace",
            "Index and middle finger raised in a V: the peace or victory sign",
        ),
        (
            "open_palm",
            "An open hand with the palm facing the camera, fingers spread",
        ),
        ("pointing", "One index finger pointing"),
        ("heart", "Both hands joined in a heart shape"),
        ("fist", "A closed fist, no finger or thumb extended"),
        ("none", "No hand gesture, or no hand visible"),
    ] {
        hand_criteria.insert(k.to_string(), Some(desc.to_string()));
    }

    let mut face_criteria = IndexMap::new();
    for (k, desc) in [
        ("smile", "Smiling or laughing"),
        ("sad", "Sad, frowning, corners of the mouth down"),
        ("surprised", "Surprised, mouth open, eyebrows raised"),
        (
            "thinking",
            "Thinking, pensive, hand on the chin or looking up",
        ),
        ("neutral", "Neutral, relaxed face"),
        ("none", "No face visible"),
    ] {
        face_criteria.insert(k.to_string(), Some(desc.to_string()));
    }

    let mut questions = IndexMap::with_capacity(4);
    questions.insert(
        "hand".to_string(),
        Question::Choice {
            instructions: "What gesture is the person making with their hand?".to_string(),
            criteria: hand_criteria,
        },
    );
    questions.insert(
        "face".to_string(),
        Question::Choice {
            instructions: "What is the person's facial expression?".to_string(),
            criteria: face_criteria,
        },
    );
    questions.insert(
        "hand_box".to_string(),
        Question::Box {
            instructions: "Where is the hand making the gesture?".to_string(),
        },
    );
    questions.insert(
        "face_box".to_string(),
        Question::Box {
            instructions: "Where is the person's face?".to_string(),
        },
    );
    questions
}
