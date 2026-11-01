use scalix_core::{
    AlignedBuffer, Backend, FilterMode, GlBackend, GlStrategy, ImageDesc, ImageDescMut,
    PixelFormat, ResizeOptions, ScalixError,
};

/// Every GL strategy x packed format must either succeed with non-black output or report the
/// GL error, never return `Ok` with an untouched destination.
#[test]
fn test_gl_strategy_format_matrix_reports_errors() {
    let backend = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let compute_supported = backend.context().is_compute_supported;

    let formats = [
        PixelFormat::Rgba8888,
        PixelFormat::Bgra8888,
        PixelFormat::Rgb888,
        PixelFormat::Bgr888,
    ];
    let strategies = [
        GlStrategy::Blit,
        GlStrategy::Raster,
        GlStrategy::LodPyramid,
        GlStrategy::Compute,
    ];
    let mut failures = Vec::new();

    for format in formats {
        let bpp = format.bytes_per_pixel().unwrap();
        let mut src = AlignedBuffer::new(128 * 128 * bpp).unwrap();
        src.as_mut_slice().fill(180);
        for strategy in strategies {
            if strategy == GlStrategy::Compute && !compute_supported {
                continue;
            }
            let mut dst = AlignedBuffer::new(32 * 32 * bpp).unwrap();
            let src_desc = ImageDesc::new(128, 128, 128 * bpp, format, &src).unwrap();
            let mut dst_desc = ImageDescMut::new(32, 32, 32 * bpp, format, &mut dst).unwrap();
            let opts = ResizeOptions::new(FilterMode::Bilinear).with_gl_strategy(strategy);
            match backend.process(&src_desc, &mut dst_desc, &opts) {
                Err(e) => failures.push(format!("{strategy:?} {format:?}: {e}")),
                Ok(()) if dst.as_slice()[0] == 0 => {
                    failures.push(format!("{strategy:?} {format:?}: Ok but output is black"))
                }
                Ok(()) => {}
            }
        }
    }
    assert!(failures.is_empty(), "GL failures:\n{}", failures.join("\n"));
}
