//! Integration tests verifying LFM3-VL + d1 System-One execution, box decoding, and HTTP endpoints.

use std::collections::HashMap;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use http_body_util::BodyExt;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use mlx_rs::ops::QuantizeOptions;
use mlx_rs::{ops, Array};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};
use tower::ServiceExt;

use mlx_d1_gestures::box_decode::{
    box_confidence, cdf_crossing, decode_many_boxes, decode_single_box, ordered_pair, NormalizedBox,
};
use mlx_d1_gestures::calibration::{interpolate_knots, Calibration, CalibrationKind};
use mlx_d1_gestures::config::{
    BoxEdgeRead, ImageProcessorConfig, LayerType, ModelConfig, PixelBounds, ProcessorConfig,
    QuantizationConfig, RopeSpec, TextConfig, TokenizerMeta, VisionConfig,
};
use mlx_d1_gestures::engine::D1Engine;
use mlx_d1_gestures::grid::{Grid, GRID_128, THOUSANDTHS_GRID};
use mlx_d1_gestures::image_proc::{preprocess_image, smart_resize};
use mlx_d1_gestures::mlx_layers::WeightStore;
use mlx_d1_gestures::prompt::PromptBuilder;
use mlx_d1_gestures::questions::{default_gesture_questions, Answer, Question, ScoreReadout};
use mlx_d1_gestures::server::{build_router, ServerHandle, StatusResponse, MAX_FRAME_BYTES};

#[test]
fn parses_real_d1_2b_mlx_8bit_config() {
    let raw = r#"{
        "architectures": ["D1Model"],
        "bos_token_id": 124894,
        "d1_attention": "set",
        "d1_codes": 2,
        "d1_numbers": true,
        "d1_set_size": 16,
        "d1_vision_max_pixels": 1638400,
        "downsample_factor": 2,
        "image_token_id": 125030,
        "model_type": "lfm3_vl",
        "projector_bias": true,
        "projector_hidden_size": 4096,
        "projector_layer_norm_eps": 1e-6,
        "quantization": { "group_size": 64, "bits": 8, "mode": "affine" },
        "text_config": {
            "conv_L_cache": 7,
            "conv_bias": false,
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 7680,
            "layer_types": ["conv", "full_attention", "sliding_attention"],
            "norm_eps": 1e-5,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "num_key_value_heads": 4,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.5, "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            },
            "sliding_window": 256,
            "tie_word_embeddings": true,
            "use_sandwich_norm": true,
            "vocab_size": 128000
        },
        "vision_config": {
            "conv_kernel_size": 3,
            "hidden_size": 1024,
            "in_channels": 3,
            "intermediate_size": 2560,
            "layer_types": ["conv", "full_attention"],
            "norm_eps": 1e-5,
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "num_key_value_heads": 8,
            "patch_size": 16,
            "rope_parameters": { "rope_theta": 1000000.0 },
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 125010,
        "vision_start_token_id": 125009
    }"#;

    let cfg = ModelConfig::from_json_str(raw).expect("valid d1-2B config");
    assert_eq!(cfg.text_config.conv_kernel_size(), 7);
    assert_eq!(
        cfg.effective_quantization(),
        Some(QuantizationConfig {
            group_size: 64,
            bits: 8
        })
    );
    assert_eq!(cfg.d1_box_edges, BoxEdgeRead::Median);
    assert_eq!(cfg.d1_prompt, 1);
}

#[test]
fn calibration_buckets_and_knot_interpolation_are_exact() {
    assert_eq!(CalibrationKind::Noul.bucket_key(), "noul");
    assert_eq!(CalibrationKind::Choice(3).bucket_key(), "choice:small");
    assert_eq!(CalibrationKind::Choice(8).bucket_key(), "choice:mid");
    assert_eq!(CalibrationKind::Choice(9).bucket_key(), "choice:large");
    assert_eq!(CalibrationKind::Choice(30).bucket_key(), "choice:inventory");

    let knots = [[0.2, 0.1], [0.6, 0.5], [0.8, 0.9]];
    assert!((interpolate_knots(0.05, &knots) - 0.1).abs() < 1e-9);
    assert!((interpolate_knots(0.4, &knots) - 0.3).abs() < 1e-9);
    assert!((interpolate_knots(0.95, &knots) - 0.9).abs() < 1e-9);

    let mut cal = Calibration {
        default: 1.5,
        ..Calibration::default()
    };
    cal.tables.insert("found:box".to_string(), knots.to_vec());
    assert!((cal.through("found:box", 0.4) - 0.3).abs() < 1e-9);
    assert!((cal.level("found:box", 0.3) - 0.4).abs() < 1e-9);
}

#[test]
fn halton_box_confidence_and_edge_decoding_behave_monotonically() {
    let grid = Grid::try_new(0.0, 1.0, 11).expect("valid 11-bin grid");
    let (a, b) = ordered_pair(0.7, 0.3, &grid);
    assert!(a < b, "crossed edges must become ordered: {a} < {b}");

    // Sharp distribution peaked around 0.2, 0.8, 0.1, 0.9 on THOUSANDTHS_GRID
    let make_peak = |idx: usize| {
        let mut v = vec![1e-6_f64; THOUSANDTHS_GRID.size()];
        v[idx] = 1.0;
        v
    };
    let presence_yes = vec![0.95, 0.05];
    let slots = vec![
        presence_yes,
        make_peak(200),
        make_peak(800),
        make_peak(100),
        make_peak(900),
    ];

    let (rect_opt, found, conf) = decode_single_box(&slots, None, BoxEdgeRead::Median);
    let rect = rect_opt.expect("box should be present");
    assert!((found - 0.95).abs() < 1e-6);
    assert!(
        conf > 0.99,
        "peaked edges should yield ~1.0 IoU confidence, got {conf}"
    );
    assert!((rect.top - 0.200).abs() < 1e-6);
    assert!((rect.bottom - 0.800).abs() < 1e-6);
    assert!((rect.left - 0.100).abs() < 1e-6);
    assert!((rect.right - 0.900).abs() < 1e-6);

    // Diffuse uniform edges should yield much lower IoU confidence
    let uniform_edge = vec![1.0; GRID_128.size()];
    let diffuse_conf = box_confidence(
        &[
            uniform_edge.clone(),
            uniform_edge.clone(),
            uniform_edge.clone(),
            uniform_edge,
        ],
        &NormalizedBox {
            top: 0.2,
            bottom: 0.8,
            left: 0.2,
            right: 0.8,
        },
    );
    assert!(
        diffuse_conf < 0.25,
        "uniform edges must have low confidence: {diffuse_conf}"
    );

    // CDF crossing on symmetric 2-bin mass
    let g2 = Grid::try_new(0.0, 1.0, 3).expect("3-bin grid");
    let c = cdf_crossing(&[0.25, 0.5, 0.25], &g2);
    assert!((c - 0.5).abs() < 1e-6);

    // Multi-box reading order sort (left, then top)
    let mut many = Vec::new();
    // Box 1 at left=0.6, Box 2 at left=0.1, Box 3 absent
    many.extend([
        vec![0.9, 0.1],
        make_peak(200),
        make_peak(500),
        make_peak(600),
        make_peak(900),
    ]);
    many.extend([
        vec![0.8, 0.2],
        make_peak(100),
        make_peak(400),
        make_peak(100),
        make_peak(300),
    ]);
    many.extend([
        vec![0.2, 0.8],
        make_peak(0),
        make_peak(100),
        make_peak(0),
        make_peak(100),
    ]);
    let (boxes, at_least, presence) = decode_many_boxes(&many, None, BoxEdgeRead::Median);
    assert_eq!(boxes.len(), 2);
    assert!(!at_least);
    assert_eq!(presence.len(), 3);
    assert!(boxes[0].r#box.left < boxes[1].r#box.left);
}

#[test]
fn smart_resize_and_patchify_match_lfm3_vl_dimensions() {
    let (h, w) = smart_resize(480, 640, 32, 65_536, 1_638_400).expect("640x480 resize");
    assert_eq!((h, w), (480, 640));

    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 480, Rgb([255, 128, 0])));
    let cfg = ImageProcessorConfig {
        size: PixelBounds {
            shortest_edge: 4096,
            longest_edge: 1_638_400,
        },
        ..ImageProcessorConfig::default()
    };
    let out = preprocess_image(&img, &cfg, 1_638_400).expect("preprocess succeeds");
    assert_eq!(out.grid_thw, [1, 30, 40]);
    assert_eq!(out.num_patches, 1200);
    assert_eq!(out.patch_dim, 3 * 2 * 16 * 16);
    assert_eq!(out.num_image_tokens, 300);
    assert_eq!(out.pixel_values.len(), 1200 * 1536);

    // Channel 0 = 255 -> (1.0 - 0.5) / 0.5 = +1.0
    assert!((out.pixel_values[0] - 1.0).abs() < 1e-5);
    // Channel 2 = 0 -> (0.0 - 0.5) / 0.5 = -1.0 (starts at index 2 * 2 * 256 = 1024)
    assert!((out.pixel_values[1024] - (-1.0)).abs() < 1e-5);
}

fn build_synthetic_engine() -> D1Engine {
    // Build a deterministic byte-level WordLevel/BPE-compatible test tokenizer
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0_u32;
    let mut insert_tok = |s: &str| -> u32 {
        if let Some(&id) = vocab.get(s) {
            return id;
        }
        let id = next_id;
        vocab.insert(s.to_string(), id);
        next_id += 1;
        id
    };

    let bos_id = insert_tok("<|startoftext|>");
    let eos_id = insert_tok("<|im_end|>");
    let _im_start_id = insert_tok("<|im_start|>");
    let vis_start_id = insert_tok("<|vision_start|>");
    let vis_end_id = insert_tok("<|vision_end|>");
    let img_tok_id = insert_tok("<|image|>");
    let _think_open = insert_tok("<think>");
    let _think_close = insert_tok("</think>");

    // Add single-token ASCII bytes/words, option codes A..Z, yes/no, colon, newline, and numbers
    for tok in [
        "yes",
        "Yes",
        "YES",
        "no",
        "No",
        "NO",
        ":",
        ":\n",
        "\n",
        " ",
        "user",
        "assistant",
        "present",
        "top",
        "bottom",
        "left",
        "right",
    ] {
        insert_tok(tok);
    }
    for c in b'A'..=b'Z' {
        insert_tok(&char::from(c).to_string());
    }
    for a in b'A'..=b'F' {
        for b in b'A'..=b'Z' {
            insert_tok(&format!("{}{}", char::from(a), char::from(b)));
        }
    }
    for n in 0..1000 {
        insert_tok(&n.to_string());
    }
    // Ensure ByteLevel alphabet tokens exist so arbitrary text encodes cleanly without UNK failure
    for b in ByteLevel::alphabet() {
        insert_tok(&b.to_string());
    }
    insert_tok("<unk>");

    // Pad vocabulary to a multiple of 64 so 8-bit group_size=64 quantization works cleanly
    while !vocab.len().is_multiple_of(64) {
        let id = vocab.len() as u32;
        vocab.insert(format!("<pad_{id}>"), id);
    }
    let vocab_size = vocab.len();

    let word_level = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .expect("wordlevel model");

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(word_level)
        .with_pre_tokenizer(Some(ByteLevel::new(false, false, false)))
        .with_post_processor(None::<tokenizers::processors::byte_level::ByteLevel>)
        .with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()))
        .with_normalizer(None::<tokenizers::normalizers::unicode::NFC>)
        .build()
        .expect("build synthetic tokenizer");

    tokenizer
        .add_special_tokens([
            AddedToken::from("<|startoftext|>", true),
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
            AddedToken::from("<|vision_start|>", true),
            AddedToken::from("<|vision_end|>", true),
            AddedToken::from("<|image|>", true),
            AddedToken::from("<think>", true),
            AddedToken::from("</think>", true),
        ])
        .expect("add special tokens");
    let mut extra_tokens = vec![
        AddedToken::from("yes", false),
        AddedToken::from("no", false),
        AddedToken::from(":", false),
        AddedToken::from("present", false),
        AddedToken::from("top", false),
        AddedToken::from("bottom", false),
        AddedToken::from("left", false),
        AddedToken::from("right", false),
    ];
    for n in 0..1000 {
        extra_tokens.push(AddedToken::from(n.to_string(), false));
    }
    let _ = tokenizer.add_tokens(extra_tokens);

    let mut rope_params = HashMap::new();
    rope_params.insert(
        "full_attention".to_string(),
        RopeSpec {
            rope_theta: 100_000.0,
            partial_rotary_factor: 0.5,
        },
    );
    rope_params.insert(
        "sliding_attention".to_string(),
        RopeSpec {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
        },
    );

    // Use dimensions divisible by 64 so 8-bit group_size=64 affine quantization is exercised end-to-end!
    let text_hidden = 64usize;
    let text_inter = 128usize;
    let vision_hidden = 64usize;
    let vision_inter = 128usize;
    let proj_hidden = 128usize;

    let config = ModelConfig {
        model_type: "lfm3_vl".to_string(),
        image_token_id: img_tok_id,
        vision_start_token_id: vis_start_id,
        vision_end_token_id: vis_end_id,
        downsample_factor: 2,
        projector_hidden_size: proj_hidden,
        projector_bias: true,
        projector_layer_norm_eps: 1e-6,
        d1_attention: "set".to_string(),
        d1_codes: 2,
        d1_numbers: true, // exercises exact d1-2B 0..999 thousandths readout
        d1_set_size: 16,
        d1_vision_max_pixels: 65_536,
        d1_box_edges: BoxEdgeRead::Interp,
        d1_prompt: 1,
        quantization: Some(QuantizationConfig {
            group_size: 64,
            bits: 8,
        }),
        quantization_config: None,
        text_config: TextConfig {
            vocab_size,
            hidden_size: text_hidden,
            intermediate_size: text_inter,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            conv_l_cache: 3,
            conv_bias: false,
            norm_eps: 1e-5,
            layer_types: vec![
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::SlidingAttention,
            ],
            rope_parameters: rope_params,
            sliding_window: 32,
            tie_word_embeddings: true,
            use_sandwich_norm: true,
        },
        vision_config: VisionConfig {
            hidden_size: vision_hidden,
            intermediate_size: vision_inter,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            in_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            conv_kernel_size: 3,
            norm_eps: 1e-5,
            layer_types: vec![LayerType::Conv, LayerType::FullAttention],
            rope_parameters: RopeSpec {
                rope_theta: 100_000.0,
                partial_rotary_factor: 1.0,
            },
        },
    };
    let _ = (bos_id, eos_id);

    let processor_config = ProcessorConfig {
        image_processor: ImageProcessorConfig {
            size: PixelBounds {
                shortest_edge: 32 * 32,
                longest_edge: 64 * 64,
            },
            ..ImageProcessorConfig::default()
        },
    };

    let prompt_builder =
        PromptBuilder::from_parts(tokenizer.into(), TokenizerMeta::default(), &config)
            .expect("prompt builder");

    // Populate synthetic tensors (dense fp32 for vision/projector, 8-bit quantized for language!)
    let mut raw: HashMap<String, Array> = HashMap::new();
    let dense = |out_d: usize, in_d: usize| -> Array {
        let mut data = vec![0.0_f32; out_d * in_d];
        for i in 0..out_d {
            for j in 0..in_d {
                data[i * in_d + j] = (((i * 31 + j * 17 + 3) % 97) as f32 - 48.0) * 0.002;
            }
        }
        Array::from_slice(&data, &[out_d as i32, in_d as i32])
    };
    let insert_qlinear =
        |map: &mut HashMap<String, Array>, prefix: &str, out_d: usize, in_d: usize| {
            let w = dense(out_d, in_d);
            let (wq, scales, biases) = ops::quantize(
                &w,
                QuantizeOptions {
                    group_size: Some(64),
                    bits: Some(8),
                    ..Default::default()
                },
            )
            .expect("quantize 8-bit");
            map.insert(format!("{prefix}.weight"), wq);
            map.insert(format!("{prefix}.scales"), scales);
            map.insert(
                format!("{prefix}.biases"),
                biases.expect("affine quantization biases"),
            );
        };
    let ones_1d = |d: usize| Array::ones::<f32>(&[d as i32]).expect("ones");
    let zeros_1d = |d: usize| Array::zeros::<f32>(&[d as i32]).expect("zeros");

    // Vision tower weights
    let patch_dim = 3 * 2 * 16 * 16;
    raw.insert(
        "vision_tower.patchifier.weight".to_string(),
        dense(vision_hidden, patch_dim),
    );
    raw.insert(
        "vision_tower.embedding_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    // Vision layer 0: Conv
    raw.insert(
        "vision_tower.layers.0.operator_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.0.ffn_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.0.conv.in_proj.weight".to_string(),
        dense(3 * vision_hidden, vision_hidden),
    );
    // PyTorch shape [C, 1, 3, 3] to verify WeightStore sanitization transposes to [C, 3, 3, 1]!
    raw.insert(
        "vision_tower.layers.0.conv.spatial_conv.weight".to_string(),
        Array::ones::<f32>(&[vision_hidden as i32, 1, 3, 3]).expect("spatial conv") * 0.05,
    );
    raw.insert(
        "vision_tower.layers.0.conv.out_proj.weight".to_string(),
        dense(vision_hidden, vision_hidden),
    );
    for w in ["w1", "w3"] {
        raw.insert(
            format!("vision_tower.layers.0.feed_forward.{w}.weight"),
            dense(vision_inter, vision_hidden),
        );
    }
    raw.insert(
        "vision_tower.layers.0.feed_forward.w2.weight".to_string(),
        dense(vision_hidden, vision_inter),
    );

    // Vision layer 1: FullAttention
    raw.insert(
        "vision_tower.layers.1.operator_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.ffn_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.q_proj.weight".to_string(),
        dense(vision_hidden, vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.k_proj.weight".to_string(),
        dense(vision_hidden / 2, vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.v_proj.weight".to_string(),
        dense(vision_hidden / 2, vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.out_proj.weight".to_string(),
        dense(vision_hidden, vision_hidden),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.q_layernorm.weight".to_string(),
        ones_1d(16),
    );
    raw.insert(
        "vision_tower.layers.1.self_attn.k_layernorm.weight".to_string(),
        ones_1d(16),
    );
    for w in ["w1", "w3"] {
        raw.insert(
            format!("vision_tower.layers.1.feed_forward.{w}.weight"),
            dense(vision_inter, vision_hidden),
        );
    }
    raw.insert(
        "vision_tower.layers.1.feed_forward.w2.weight".to_string(),
        dense(vision_hidden, vision_inter),
    );

    // Multimodal projector
    raw.insert(
        "multi_modal_projector.layer_norm.weight".to_string(),
        ones_1d(vision_hidden),
    );
    raw.insert(
        "multi_modal_projector.layer_norm.bias".to_string(),
        zeros_1d(vision_hidden),
    );
    raw.insert(
        "multi_modal_projector.linear_1.weight".to_string(),
        dense(proj_hidden, vision_hidden * 4),
    );
    raw.insert(
        "multi_modal_projector.linear_1.bias".to_string(),
        zeros_1d(proj_hidden),
    );
    raw.insert(
        "multi_modal_projector.linear_2.weight".to_string(),
        dense(text_hidden, proj_hidden),
    );
    raw.insert(
        "multi_modal_projector.linear_2.bias".to_string(),
        zeros_1d(text_hidden),
    );

    // 8-bit Quantized Language Backbone (`model.language_model.*` prefix to also test sanitize!)
    insert_qlinear(
        &mut raw,
        "model.language_model.embed_tokens",
        vocab_size,
        text_hidden,
    );
    raw.insert(
        "model.language_model.embedding_norm.weight".to_string(),
        ones_1d(text_hidden),
    );

    for layer_idx in 0..3 {
        let lp = format!("model.language_model.layers.{layer_idx}");
        for norm in [
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ] {
            raw.insert(format!("{lp}.{norm}.weight"), ones_1d(text_hidden));
        }
        insert_qlinear(
            &mut raw,
            &format!("{lp}.feed_forward.w1"),
            text_inter,
            text_hidden,
        );
        insert_qlinear(
            &mut raw,
            &format!("{lp}.feed_forward.w3"),
            text_inter,
            text_hidden,
        );
        insert_qlinear(
            &mut raw,
            &format!("{lp}.feed_forward.w2"),
            text_hidden,
            text_inter,
        );

        if layer_idx == 0 {
            insert_qlinear(
                &mut raw,
                &format!("{lp}.conv.in_proj"),
                3 * text_hidden,
                text_hidden,
            );
            insert_qlinear(
                &mut raw,
                &format!("{lp}.conv.out_proj"),
                text_hidden,
                text_hidden,
            );
            // PyTorch [D, 1, K] shape -> WeightStore transposes to [D, K, 1]
            raw.insert(
                format!("{lp}.conv.conv.weight"),
                Array::ones::<f32>(&[text_hidden as i32, 1, 3]).expect("conv1d") * 0.1,
            );
        } else {
            insert_qlinear(&mut raw, &format!("{lp}.self_attn.q_proj"), 64, text_hidden);
            insert_qlinear(&mut raw, &format!("{lp}.self_attn.k_proj"), 32, text_hidden);
            insert_qlinear(&mut raw, &format!("{lp}.self_attn.v_proj"), 32, text_hidden);
            insert_qlinear(&mut raw, &format!("{lp}.self_attn.o_proj"), text_hidden, 64);
            // gate_proj out_features = 4 (smaller than group_size 64 -> dense linear fallback)
            raw.insert(
                format!("{lp}.self_attn.gate_proj.weight"),
                dense(4, text_hidden),
            );
            raw.insert(format!("{lp}.self_attn.q_norm.weight"), ones_1d(16));
            raw.insert(format!("{lp}.self_attn.k_norm.weight"), ones_1d(16));
        }
    }

    let mut store = WeightStore::from_raw_tensors(
        raw,
        config.effective_quantization(),
        config.text_config.tie_word_embeddings,
    )
    .expect("sanitize weights");

    D1Engine::from_parts(
        "synthetic/d1-tiny-8bit".to_string(),
        config,
        processor_config,
        prompt_builder,
        Some(Calibration {
            default: 1.337,
            ..Calibration::default()
        }),
        &mut store,
    )
    .expect("construct synthetic D1Engine")
}

#[tokio::test]
async fn end_to_end_metal_forward_pass_and_http_routes_succeed() {
    let questions = default_gesture_questions();

    // 1. Spawn ServerHandle (constructs engine & runs warmup pass inside worker thread)
    let (handle, warm) = ServerHandle::try_spawn(questions, || Ok(build_synthetic_engine()))
        .expect("spawn & warmup");
    assert_eq!(warm.usage.output_tokens, 0);
    assert!(warm.usage.input_tokens > 20);
    assert!(warm.answers.contains_key("hand"));
    assert!(warm.answers.contains_key("face"));
    assert!(warm.answers.contains_key("hand_box"));
    assert!(warm.answers.contains_key("face_box"));

    if let Some(Answer::Choice { probabilities, .. }) = warm.answers.get("hand") {
        let sum: f64 = probabilities.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "hand probabilities must sum to 1, got {sum}"
        );
        assert_eq!(probabilities.len(), 9);
    } else {
        panic!("expected Choice answer for `hand`");
    }

    let app = build_router(handle);

    // GET /status
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let status: StatusResponse = serde_json::from_slice(&body).unwrap();
    assert!(status.ready);
    assert_eq!(status.model, "synthetic/d1-tiny-8bit");

    // GET /
    let index_resp = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(index_resp.status(), StatusCode::OK);

    // POST /run with valid JPEG frame
    let mut jpeg_bytes = Vec::new();
    let test_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([40, 120, 200])));
    test_img
        .write_to(
            &mut std::io::Cursor::new(&mut jpeg_bytes),
            ImageFormat::Jpeg,
        )
        .unwrap();

    let run_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/run")
                .header(header::CONTENT_TYPE, "image/jpeg")
                .body(Body::from(jpeg_bytes))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(run_resp.status(), StatusCode::OK);
    let run_body = run_resp.into_body().collect().await.unwrap().to_bytes();
    let decision: mlx_d1_gestures::DecisionResponse = serde_json::from_slice(&run_body).unwrap();
    assert_eq!(decision.usage.output_tokens, 0);
    assert_eq!(decision.answers.len(), 4);

    // POST /run with empty body -> 413 matching `decide.py`
    let empty_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/run")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty_resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

    // POST /run with > 4 MB Content-Length -> 413
    let huge_resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/run")
                .header(header::CONTENT_LENGTH, (MAX_FRAME_BYTES + 10).to_string())
                .body(Body::from(vec![0_u8; 16]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(huge_resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[test]
fn question_json_round_trip_supports_score_and_estimate() {
    let score_json =
        r#"{"type":"score","instructions":"Rate clarity","criteria":["Low","Medium","High"]}"#;
    let q: Question = serde_json::from_str(score_json).unwrap();
    assert!(!q.is_typed());
    assert_eq!(ScoreReadout::default(), ScoreReadout::Median);

    let est_json = r#"{"type":"number","instructions":"How many fingers?","range":[0,5],"step":1,"unit":"fingers"}"#;
    let est: Question = serde_json::from_str(est_json).unwrap();
    assert!(est.is_typed());
    let grid = est.estimate_grid().unwrap();
    assert_eq!(grid.size(), 6);
}
