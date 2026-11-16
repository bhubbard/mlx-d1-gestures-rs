//! Pure-Rust LFM3-VL image downscaling, `smart_resize`, normalization, and spatial-merge patchification.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

use crate::config::ImageProcessorConfig;
use crate::error::{D1Error, Result};

const MAX_ASPECT_RATIO: f64 = 200.0;

/// Patchified image tensor data and `(T, H, W)` patch grid metadata ready for MLX upload.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedImage {
    /// Flattened patch tensor `[num_patches, patch_dim]` in row-major `f32`.
    pub pixel_values: Vec<f32>,
    /// Number of spatial patches (`grid_h * grid_w`).
    pub num_patches: usize,
    /// Feature dimension per patch (`3 * temporal_patch_size * patch_size * patch_size` = `1536`).
    pub patch_dim: usize,
    /// Vision tower grid `[temporal=1, grid_h, grid_w]`.
    pub grid_thw: [usize; 3],
    /// Number of `<|image|>` placeholder tokens (`num_patches / merge_size^2`).
    pub num_image_tokens: usize,
}

/// Downscales `image` with bicubic resampling when `width * height > max_pixels`.
#[must_use]
pub fn cap_pixels(image: &DynamicImage, max_pixels: usize) -> RgbImage {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let total = (w as usize) * (h as usize);
    if total <= max_pixels || max_pixels == 0 {
        return rgb;
    }
    let scale = ((max_pixels as f64) / (total as f64)).sqrt();
    let target_w = ((f64::from(w) * scale).floor() as u32).max(1);
    let target_h = ((f64::from(h) * scale).floor() as u32).max(1);
    image::imageops::resize(&rgb, target_w, target_h, FilterType::CatmullRom)
}

/// Computes target `(height, width)` divisible by `factor` within `[min_pixels, max_pixels]`.
///
/// # Errors
/// Returns [`D1Error::InvalidRequest`] if either dimension is zero or aspect ratio exceeds `200`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(D1Error::InvalidRequest(
            "image dimensions and resize factor must be non-zero".to_string(),
        ));
    }
    let ratio = (height.max(width) as f64) / (height.min(width) as f64);
    if ratio > MAX_ASPECT_RATIO {
        return Err(D1Error::InvalidRequest(format!(
            "image aspect ratio {ratio:.1} exceeds {MAX_ASPECT_RATIO}"
        )));
    }

    let factor_f = factor as f64;
    let h_f = height as f64;
    let w_f = width as f64;

    let mut h_bar = ((h_f / factor_f).round() as usize).max(1) * factor;
    let mut w_bar = ((w_f / factor_f).round() as usize).max(1) * factor;

    if h_bar * w_bar > max_pixels {
        let beta = ((h_f * w_f) / (max_pixels as f64)).sqrt();
        h_bar = (((h_f / beta / factor_f).floor() as usize) * factor).max(factor);
        w_bar = (((w_f / beta / factor_f).floor() as usize) * factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = ((min_pixels as f64) / (h_f * w_f)).sqrt();
        h_bar = ((h_f * beta / factor_f).ceil() as usize) * factor;
        w_bar = ((w_f * beta / factor_f).ceil() as usize) * factor;
    }

    Ok((h_bar, w_bar))
}

/// Preprocesses an input [`DynamicImage`] into LFM3-VL's merged spatial-temporal patch layout.
///
/// # Errors
/// Returns [`D1Error`] if `smart_resize` fails on extreme aspect ratios or zero dimensions.
pub fn preprocess_image(
    image: &DynamicImage,
    proc_cfg: &ImageProcessorConfig,
    d1_max_pixels: usize,
) -> Result<ProcessedImage> {
    let capped = cap_pixels(image, d1_max_pixels);
    let (cur_w, cur_h) = capped.dimensions();
    let patch = proc_cfg.patch_size;
    let merge = proc_cfg.merge_size;
    let temporal = proc_cfg.temporal_patch_size;
    let factor = patch * merge;

    let resized = if proc_cfg.do_resize {
        let (target_h, target_w) = smart_resize(
            cur_h as usize,
            cur_w as usize,
            factor,
            proc_cfg.size.shortest_edge,
            proc_cfg.size.longest_edge,
        )?;
        if target_h == cur_h as usize && target_w == cur_w as usize {
            capped
        } else {
            image::imageops::resize(
                &capped,
                target_w as u32,
                target_h as u32,
                FilterType::CatmullRom,
            )
        }
    } else {
        capped
    };

    let (width_u32, height_u32) = resized.dimensions();
    let height = height_u32 as usize;
    let width = width_u32 as usize;
    if !height.is_multiple_of(factor) || !width.is_multiple_of(factor) {
        return Err(D1Error::InvalidRequest(format!(
            "resized frame {width}x{height} is not divisible by factor {factor}"
        )));
    }

    let grid_h = height / patch;
    let grid_w = width / patch;
    let num_patches = grid_h * grid_w;
    let patch_dim = 3 * temporal * patch * patch;
    let raw_bytes = resized.as_raw();

    let scale = if proc_cfg.do_rescale {
        proc_cfg.rescale_factor
    } else {
        1.0
    };
    let mean = if proc_cfg.do_normalize {
        proc_cfg.image_mean
    } else {
        [0.0, 0.0, 0.0]
    };
    let inv_std = if proc_cfg.do_normalize {
        [
            1.0 / proc_cfg.image_std[0],
            1.0 / proc_cfg.image_std[1],
            1.0 / proc_cfg.image_std[2],
        ]
    } else {
        [1.0, 1.0, 1.0]
    };

    // Exact permutation of `Lfm3VlImageProcessor.patchify`:
    // Outer patch order: (gh_block in 0..grid_h/merge, gw_block in 0..grid_w/merge, mh in 0..merge, mw in 0..merge)
    // Inner patch vector: (c in 0..3, t in 0..temporal, py in 0..patch, px in 0..patch)
    let mut pixel_values = Vec::with_capacity(num_patches * patch_dim);
    let single_frame_plane = patch * patch;

    for gh_block in 0..(grid_h / merge) {
        for gw_block in 0..(grid_w / merge) {
            for mh in 0..merge {
                for mw in 0..merge {
                    let base_y = (gh_block * merge + mh) * patch;
                    let base_x = (gw_block * merge + mw) * patch;

                    for c in 0..3 {
                        let m = mean[c];
                        let inv_s = inv_std[c];
                        let mut plane = Vec::with_capacity(single_frame_plane);
                        for py in 0..patch {
                            let row_offset = ((base_y + py) * width + base_x) * 3;
                            for px in 0..patch {
                                let byte = raw_bytes[row_offset + px * 3 + c];
                                let val = (f32::from(byte) * scale - m) * inv_s;
                                plane.push(val);
                            }
                        }
                        for _ in 0..temporal {
                            pixel_values.extend_from_slice(&plane);
                        }
                    }
                }
            }
        }
    }

    Ok(ProcessedImage {
        pixel_values,
        num_patches,
        patch_dim,
        grid_thw: [1, grid_h, grid_w],
        num_image_tokens: num_patches / (merge * merge),
    })
}
