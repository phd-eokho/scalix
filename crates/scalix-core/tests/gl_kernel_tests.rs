use scalix_core::{
    AlignedBuffer, Backend, FilterMode, GlBackend, GlStrategy, ImageDesc, ImageDescMut,
    PixelFormat, ResizeOptions, ScalixError,
};

/// 8-bit rounding plus GPU float precision.
const MAX_CHANNEL_ERROR: i32 = 2;

fn catmull_rom(f: f64) -> [f64; 4] {
    let (f2, f3) = (f * f, f * f * f);
    [
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    ]
}

fn lanczos3(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-5 {
        return 1.0;
    }
    if ax >= 3.0 {
        return 0.0;
    }
    let pix = std::f64::consts::PI * ax;
    (pix.sin() * (pix / 3.0).sin() * 3.0) / (pix * pix)
}

/// Separable taps `(offset_from_floor, weight)` for one axis.
fn taps(filter: FilterMode, f: f64) -> Vec<(i64, f64)> {
    match filter {
        FilterMode::Bicubic => catmull_rom(f)
            .iter()
            .enumerate()
            .map(|(i, &w)| (i as i64 - 1, w))
            .collect(),
        FilterMode::Lanczos3 => {
            let w: Vec<f64> = (0..6).map(|i| lanczos3((i - 2) as f64 - f)).collect();
            let sum: f64 = w.iter().sum();
            w.iter()
                .enumerate()
                .map(|(i, &wi)| (i as i64 - 2, wi / sum.abs().max(1e-5)))
                .collect()
        }
        other => unreachable!("{other:?}"),
    }
}

/// CPU reference matching the documented convention `src = (dst + 0.5) * scale - 0.5`.
fn reference(
    filter: FilterMode,
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<u8> {
    let (sx, sy) = (sw as f64 / dw as f64, sh as f64 / dh as f64);
    let mut out = vec![0u8; dw * dh * 4];
    for dy in 0..dh {
        let v = (dy as f64 + 0.5) * sy - 0.5;
        let ty = taps(filter, v - v.floor());
        for dx in 0..dw {
            let u = (dx as f64 + 0.5) * sx - 0.5;
            let tx = taps(filter, u - u.floor());
            for c in 0..4 {
                let mut acc = 0.0;
                for &(oy, wy) in &ty {
                    let y = (v.floor() as i64 + oy).clamp(0, sh as i64 - 1) as usize;
                    for &(ox, wx) in &tx {
                        let x = (u.floor() as i64 + ox).clamp(0, sw as i64 - 1) as usize;
                        acc += wy * wx * src[(y * sw + x) * 4 + c] as f64 / 255.0;
                    }
                }
                out[(dy * dw + dx) * 4 + c] = (acc.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

#[test]
fn test_gl_bicubic_lanczos3_match_reference() {
    let backend = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let format = PixelFormat::Rgba8888;
    let (sw, sh) = (48usize, 40usize);
    let mut src = AlignedBuffer::new(sw * sh * 4).unwrap();
    for (i, px) in src.as_mut_slice().chunks_mut(4).enumerate() {
        let (x, y) = (i % sw, i / sw);
        // High-frequency checker + gradients: exposes kernel shape differences.
        let checker = if (x / 3 + y / 3) % 2 == 0 { 230 } else { 20 };
        px.copy_from_slice(&[checker, (x * 5) as u8, (y * 6) as u8, 255]);
    }

    let mut strategies = vec![GlStrategy::Raster];
    if backend.context().is_compute_supported {
        strategies.push(GlStrategy::Compute);
    }

    for filter in [FilterMode::Bicubic, FilterMode::Lanczos3] {
        for (dw, dh) in [(20usize, 17usize), (75, 61)] {
            let expected = reference(filter, src.as_slice(), sw, sh, dw, dh);
            for &strategy in &strategies {
                let mut dst = AlignedBuffer::new(dw * dh * 4).unwrap();
                let src_desc = ImageDesc::new(sw as u32, sh as u32, sw * 4, format, &src).unwrap();
                let mut dst_desc =
                    ImageDescMut::new(dw as u32, dh as u32, dw * 4, format, &mut dst).unwrap();
                backend
                    .process(
                        &src_desc,
                        &mut dst_desc,
                        &ResizeOptions::new(filter).with_gl_strategy(strategy),
                    )
                    .unwrap_or_else(|e| panic!("{strategy:?} {filter:?}: {e:?}"));
                let max_err = dst
                    .as_slice()
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (a as i32 - b as i32).abs())
                    .max()
                    .unwrap();
                assert!(
                    max_err <= MAX_CHANNEL_ERROR,
                    "{strategy:?} {filter:?} {sw}x{sh}->{dw}x{dh}: max channel error {max_err}"
                );
            }
        }
    }
}
