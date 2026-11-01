use scalix_core::{
    AlignedBuffer, Backend, FilterMode, ImageDesc, ImageDescMut, PixelFormat, ResizeOptions,
    VulkanBackend, VulkanStrategy,
};

const SRC: u32 = 64;
const DST: u32 = 32;
const ITERATIONS: usize = 40;
const THREADS_PER_STRATEGY: usize = 2;

/// Every strategy of one backend shares the device queue. Running them from several threads at
/// once must neither fail nor mix up results: each thread uses a uniform source with its own
/// value, so any cross-talk or lost submission shows up as a wrong destination byte.
#[test]
fn test_vulkan_strategies_run_concurrently_on_one_backend() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Vulkan backend not available ({e:?}); skipping");
            return;
        }
    };
    let strategies = [
        VulkanStrategy::Blit,
        VulkanStrategy::Raster,
        VulkanStrategy::LodPyramid,
        VulkanStrategy::Compute,
    ];

    std::thread::scope(|scope| {
        for (i, &strategy) in strategies
            .iter()
            .cycle()
            .take(strategies.len() * THREADS_PER_STRATEGY)
            .enumerate()
        {
            let backend = &backend;
            scope.spawn(move || run_worker(backend, strategy, 16 + 24 * i as u8));
        }
    });
}

fn run_worker(backend: &VulkanBackend, strategy: VulkanStrategy, value: u8) {
    let format = PixelFormat::Rgba8888;
    let src_stride = format.min_stride(SRC).unwrap();
    let dst_stride = format.min_stride(DST).unwrap();
    let mut src = AlignedBuffer::new(src_stride * SRC as usize).unwrap();
    src.as_mut_slice().fill(value);
    let mut dst = AlignedBuffer::new(dst_stride * DST as usize).unwrap();
    let opts = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(strategy);

    for iter in 0..ITERATIONS {
        dst.as_mut_slice().fill(0);
        {
            let src_desc = ImageDesc::new(SRC, SRC, src_stride, format, &src).unwrap();
            let mut dst_desc = ImageDescMut::new(DST, DST, dst_stride, format, &mut dst).unwrap();
            if let Err(e) = backend.process(&src_desc, &mut dst_desc, &opts) {
                panic!("{strategy:?} iteration {iter} failed: {e:?}");
            }
        }
        if let Some(pos) = dst.as_slice().iter().position(|&b| b != value) {
            panic!(
                "{strategy:?} iteration {iter}: byte {pos} = {} expected {value}",
                dst.as_slice()[pos]
            );
        }
    }
}
