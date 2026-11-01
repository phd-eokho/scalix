use scalix_core::{
    AlignedBuffer, Backend, FilterMode, GlBackend, GlStrategy, ImageDesc, ImageDescMut,
    PixelFormat, ResizeOptions, ScalixError,
};

/// Dropping one GL backend must not invalidate a sibling sharing the same EGLDisplay.
#[test]
fn test_gl_sibling_backend_survives_drop() {
    let first = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let second = GlBackend::new().expect("second GlBackend");
    drop(first);

    let format = PixelFormat::Rgba8888;
    let mut src = AlignedBuffer::new(64 * 64 * 4).unwrap();
    src.as_mut_slice().fill(200);
    let mut dst = AlignedBuffer::new(32 * 32 * 4).unwrap();
    let src_desc = ImageDesc::new(64, 64, 64 * 4, format, &src).unwrap();
    let mut dst_desc = ImageDescMut::new(32, 32, 32 * 4, format, &mut dst).unwrap();

    let opts = ResizeOptions::new(FilterMode::Bilinear).with_gl_strategy(GlStrategy::Blit);
    second
        .process(&src_desc, &mut dst_desc, &opts)
        .expect("sibling GlBackend must keep working after another is dropped");
    assert_eq!(dst.as_slice()[0], 200);
}
