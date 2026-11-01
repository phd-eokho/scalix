use scalix_core::{
    AlignedBuffer, Backend, FilterMode, GlBackend, GlStrategy, ImageDesc, ImageDescMut,
    PixelFormat, ResizeOptions, ScalixError,
};

const W: u32 = 64;
const H: u32 = 64;
const PAD_BYTE: u8 = 0xEE;

fn gl_backend() -> Option<GlBackend> {
    match GlBackend::new() {
        Ok(b) => Some(b),
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            None
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    }
}

/// Identity-size resize with padded strides must reproduce every pixel row exactly and
/// leave destination padding untouched.
fn assert_identity_with_stride(
    backend: &GlBackend,
    format: PixelFormat,
    src_stride: usize,
    dst_stride: usize,
    strategy: GlStrategy,
) {
    let bpp = format.bytes_per_pixel().unwrap();
    let row_bytes = W as usize * bpp;

    let mut src = AlignedBuffer::new(src_stride * H as usize).unwrap();
    for (y, row) in src.as_mut_slice().chunks_mut(src_stride).enumerate() {
        for (x, px) in row[..row_bytes].chunks_mut(bpp).enumerate() {
            px.fill(((x * 3 + y * 5) % 251) as u8);
        }
        row[row_bytes..].fill(PAD_BYTE);
    }
    let mut dst = AlignedBuffer::new(dst_stride * H as usize).unwrap();
    dst.as_mut_slice().fill(PAD_BYTE);

    {
        let src_desc = ImageDesc::new(W, H, src_stride, format, &src).unwrap();
        let mut dst_desc = ImageDescMut::new(W, H, dst_stride, format, &mut dst).unwrap();
        let opts = ResizeOptions::new(FilterMode::Nearest).with_gl_strategy(strategy);
        if let Err(e) = backend.process(&src_desc, &mut dst_desc, &opts) {
            if strategy == GlStrategy::Compute && !backend.context().is_compute_supported {
                return;
            }
            panic!("{strategy:?} {format:?} failed: {e:?}");
        }
    }

    for y in 0..H as usize {
        let s = &src.as_slice()[y * src_stride..y * src_stride + row_bytes];
        let d = &dst.as_slice()[y * dst_stride..y * dst_stride + dst_stride];
        assert_eq!(
            &d[..row_bytes],
            s,
            "{strategy:?} {format:?}: row {y} mismatch"
        );
        assert!(
            d[row_bytes..].iter().all(|&b| b == PAD_BYTE),
            "{strategy:?} {format:?}: row {y} padding overwritten"
        );
    }
}

#[test]
fn test_gl_padded_stride_rgba_all_strategies() {
    let Some(backend) = gl_backend() else { return };
    // Strides are whole pixels: exercises GL_UNPACK/PACK_ROW_LENGTH.
    for strategy in [
        GlStrategy::Blit,
        GlStrategy::Raster,
        GlStrategy::LodPyramid,
        GlStrategy::Compute,
    ] {
        assert_identity_with_stride(&backend, PixelFormat::Rgba8888, 320, 384, strategy);
    }
}

#[test]
fn test_gl_padded_stride_rgb_non_pixel_multiple() {
    let Some(backend) = gl_backend() else { return };
    // 64 px * 3 B = 192 B rows; 64-byte aligned stride 256 is not a multiple of 3 B:
    // exercises the per-row fallback.
    for strategy in [GlStrategy::Blit, GlStrategy::Raster, GlStrategy::LodPyramid] {
        assert_identity_with_stride(&backend, PixelFormat::Rgb888, 256, 200, strategy);
    }
}
