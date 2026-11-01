use scalix_core::{
    AlignedBuffer, Backend, FilterMode, GlBackend, GlStrategy, ImageDesc, ImageDescMut,
    PixelFormat, ResizeOptions, ScalixError,
};

/// 4x reduction samples mip level 2 at its native size, so every output pixel must equal the
/// average of its 4x4 source block.
#[test]
fn test_gl_lod_level2_equals_block_average() {
    const SRC: usize = 128;
    const DST: usize = 32;
    const BLOCK: usize = SRC / DST;
    const MAX_CHANNEL_ERROR: i32 = 1;

    let backend = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let format = PixelFormat::Rgba8888;
    let value = |x: usize, y: usize| ((x * 7 + y * 13) % 256) as u8;

    let mut src = AlignedBuffer::new(SRC * SRC * 4).unwrap();
    for (i, px) in src.as_mut_slice().chunks_mut(4).enumerate() {
        let v = value(i % SRC, i / SRC);
        px.copy_from_slice(&[v, 255 - v, v / 2, 255]);
    }
    let mut dst = AlignedBuffer::new(DST * DST * 4).unwrap();
    {
        let src_desc = ImageDesc::new(SRC as u32, SRC as u32, SRC * 4, format, &src).unwrap();
        let mut dst_desc =
            ImageDescMut::new(DST as u32, DST as u32, DST * 4, format, &mut dst).unwrap();
        let opts =
            ResizeOptions::new(FilterMode::Bilinear).with_gl_strategy(GlStrategy::LodPyramid);
        backend.process(&src_desc, &mut dst_desc, &opts).unwrap();
    }

    for dy in 0..DST {
        for dx in 0..DST {
            let mut sum = [0u32; 3];
            for y in dy * BLOCK..(dy + 1) * BLOCK {
                for x in dx * BLOCK..(dx + 1) * BLOCK {
                    let v = value(x, y) as u32;
                    sum[0] += v;
                    sum[1] += 255 - v;
                    sum[2] += v / 2;
                }
            }
            let n = (BLOCK * BLOCK) as f64;
            let px = &dst.as_slice()[(dy * DST + dx) * 4..][..3];
            for c in 0..3 {
                let expected = (sum[c] as f64 / n).round() as i32;
                let err = (px[c] as i32 - expected).abs();
                assert!(
                    err <= MAX_CHANNEL_ERROR,
                    "pixel ({dx},{dy}) ch{c}: got {} expected {expected}",
                    px[c]
                );
            }
        }
    }
}
