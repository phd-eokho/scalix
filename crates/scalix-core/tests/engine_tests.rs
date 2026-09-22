use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use scalix_core::{
    AlignedBuffer, Backend, BackendType, Engine, EngineConfig, FilterMode, ImageDesc, ImageDescMut,
    ImageDimensions, ImageGeometry, OwnedImage, PixelFormat, ProfileMetrics, Profiler, ScalixError,
    WorkerPool,
};

#[test]
fn test_sync_passthrough_memory_transfer() {
    let engine = Engine::new(BackendType::Passthrough).expect("Failed to create engine");

    let width = 64;
    let height = 64;
    let format = PixelFormat::Rgba8888;
    let stride = format.min_stride(width).unwrap();

    let mut src_data = AlignedBuffer::new(stride * (height as usize)).unwrap();
    // Fill with pattern
    for (i, byte) in src_data.as_mut_slice().iter_mut().enumerate() {
        *byte = (i % 255) as u8;
    }

    let mut dst_data = AlignedBuffer::new(stride * (height as usize)).unwrap();

    let src_desc = ImageDesc::new(width, height, stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(width, height, stride, format, &mut dst_data).unwrap();

    engine
        .resize_sync(&src_desc, &mut dst_desc, FilterMode::Passthrough)
        .expect("Resize sync failed");

    assert_eq!(
        src_data.as_slice(),
        dst_data.as_slice(),
        "Destination data must match source data"
    );
}

#[test]
fn test_async_task_execution() {
    let engine = Engine::new(BackendType::Auto).expect("Failed to create engine");

    let width = 32;
    let height = 32;
    let format = PixelFormat::Rgb888;

    let mut src_image = OwnedImage::allocate(width, height, format).unwrap();
    for (i, byte) in src_image.data.iter_mut().enumerate() {
        *byte = (i & 0xFF) as u8;
    }
    let expected_data = src_image.data.clone();

    let dst_image = OwnedImage::allocate(width, height, format).unwrap();

    let task = engine.resize_async(src_image, dst_image, FilterMode::Passthrough);

    let result = task
        .wait(Some(Duration::from_secs(2)))
        .expect("Task failed");
    assert_eq!(result.data, expected_data);
}

#[test]
fn test_callback_execution() {
    let engine = Engine::new(BackendType::Auto).expect("Failed to create engine");

    let width = 16;
    let height = 16;
    let format = PixelFormat::Rgba8888;

    let mut src = OwnedImage::allocate(width, height, format).unwrap();
    src.data.fill(0xAA);
    let dst = OwnedImage::allocate(width, height, format).unwrap();

    let called = Arc::new(AtomicBool::new(false));
    let called_clone = Arc::clone(&called);

    let (tx, rx) = std::sync::mpsc::channel();

    engine
        .resize_callback(src, dst, FilterMode::Passthrough, move |res| {
            called_clone.store(true, Ordering::Release);
            let _ = tx.send(res);
        })
        .expect("Callback submission failed");

    let res = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("Callback timed out");
    let completed_image = res.expect("Execution error in callback");

    assert!(called.load(Ordering::Acquire));
    assert_eq!(completed_image.data[0], 0xAA);
}

#[test]
fn test_buffer_bounds_and_error_handling() {
    // 1. Zero dimension error
    let zero_res = PixelFormat::Rgba8888.min_buffer_size(0, 100, 100);
    assert!(matches!(
        zero_res,
        Err(ScalixError::InvalidDimensions { .. })
    ));

    // 2. Undersized stride error
    let stride_res = PixelFormat::Rgba8888.min_buffer_size(100, 100, 100); // 100 < 400
    assert!(matches!(stride_res, Err(ScalixError::InvalidStride { .. })));

    // 3. Buffer too small error
    let raw_data = AlignedBuffer::new(10).unwrap();
    let img_res = ImageDesc::new(100, 100, 400, PixelFormat::Rgba8888, &raw_data);
    assert!(matches!(img_res, Err(ScalixError::BufferTooSmall { .. })));

    // 4. Unaligned pointer error
    let unaligned_raw = [0u8; 128];
    // Find unaligned byte offset
    let unaligned_offset = if (unaligned_raw.as_ptr() as usize).is_multiple_of(64) {
        1
    } else {
        0
    };
    let unaligned_slice = &unaligned_raw[unaligned_offset..unaligned_offset + 64];
    let unaligned_res = ImageDesc::new(4, 4, 16, PixelFormat::Rgba8888, unaligned_slice);
    assert!(matches!(
        unaligned_res,
        Err(ScalixError::UnalignedPointer { alignment: 64, .. })
    ));
}

#[test]
fn test_parallel_callback_dispatch() {
    use std::collections::HashSet;
    use std::sync::Mutex;

    let engine = Engine::new(BackendType::Auto).expect("Failed to create engine");

    let num_tasks = 12;
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let thread_ids = Arc::new(Mutex::new(HashSet::new()));
    let (tx, rx) = std::sync::mpsc::channel();

    for i in 0..num_tasks {
        let mut src = OwnedImage::allocate(8, 8, PixelFormat::Rgba8888).unwrap();
        src.data[0] = i as u8;
        let dst = OwnedImage::allocate(8, 8, PixelFormat::Rgba8888).unwrap();

        let count_clone = Arc::clone(&completed_count);
        let threads_clone = Arc::clone(&thread_ids);
        let tx_clone = tx.clone();

        engine
            .resize_callback(src, dst, FilterMode::Passthrough, move |res| {
                let img = res.expect("Resize failed");
                // Record the thread executing this callback
                {
                    let mut lock = threads_clone.lock().unwrap();
                    lock.insert(std::thread::current().id());
                }
                // Simulate small post-processing work
                std::thread::sleep(Duration::from_millis(15));

                let prev = count_clone.fetch_add(1, Ordering::SeqCst);
                if prev + 1 == num_tasks {
                    let _ = tx_clone.send(());
                }
                assert_eq!(img.data[0], i as u8);
            })
            .expect("Failed to submit callback");
    }

    rx.recv_timeout(Duration::from_secs(3))
        .expect("Parallel callbacks timed out");

    assert_eq!(completed_count.load(Ordering::SeqCst), num_tasks);
    let captured_threads = thread_ids.lock().unwrap().len();
    assert!(
        captured_threads > 1,
        "Callbacks should span multiple worker threads (captured {} threads)",
        captured_threads
    );
}

#[test]
fn test_thread_names_format_and_length_limit() {
    let pid = std::process::id();

    // 1. Default PID prefix: <pid>/scx-hw and <pid>/scx-w<id>
    {
        let engine = Engine::new(BackendType::Auto).expect("Failed to create engine");
        let expected_hw_name = format!("{}/scx-hw", pid);
        let expected_cb_prefix = format!("{}/scx-w", pid);

        let src = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let dst = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let task = engine.resize_async(src, dst, FilterMode::Passthrough);
        let _ = task.wait(Some(Duration::from_secs(1))).unwrap();

        let src2 = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let dst2 = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let (tx_cb, rx_cb) = std::sync::mpsc::channel();
        engine
            .resize_callback(src2, dst2, FilterMode::Passthrough, move |_| {
                let current = std::thread::current();
                let name = current.name().map(|s| s.to_string()).unwrap_or_default();
                let _ = tx_cb.send(name);
            })
            .unwrap();

        let cb_name = rx_cb.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(
            expected_hw_name.len() <= 15,
            "HW thread name '{}' exceeds Linux 15-char limit (len={})",
            expected_hw_name,
            expected_hw_name.len()
        );
        assert!(
            cb_name.len() <= 15,
            "CB thread name '{}' exceeds Linux 15-char limit (len={})",
            cb_name,
            cb_name.len()
        );
        assert!(
            cb_name.starts_with(&expected_cb_prefix),
            "CB thread name '{}' should start with '{}'",
            cb_name,
            expected_cb_prefix
        );
    }

    // 2. Custom Prefix with truncation: "cam_pipeline" -> truncated to 7 chars: "cam_pip"
    {
        let engine = Engine::with_prefix(BackendType::Auto, Some("cam_pipeline")).unwrap();
        let expected_hw_name = "cam_pip/scx-hw";
        let expected_cb_prefix = "cam_pip/scx-w";

        assert_eq!(expected_hw_name.len(), 14); // <= 15

        let src = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let dst = OwnedImage::allocate(4, 4, PixelFormat::Rgba8888).unwrap();
        let (tx_cb, rx_cb) = std::sync::mpsc::channel();
        engine
            .resize_callback(src, dst, FilterMode::Passthrough, move |_| {
                let current = std::thread::current();
                let name = current.name().map(|s| s.to_string()).unwrap_or_default();
                let _ = tx_cb.send(name);
            })
            .unwrap();

        let cb_name = rx_cb.recv_timeout(Duration::from_secs(1)).unwrap();

        assert!(
            cb_name.starts_with(expected_cb_prefix),
            "CB thread name '{}' should start with '{}'",
            cb_name,
            expected_cb_prefix
        );
    }
}

#[test]
fn test_dma_buffer_lifecycle_and_probing() {
    use scalix_core::DmaBuffer;

    let width = 64;
    let height = 64;
    let format = PixelFormat::Rgba8888;

    match DmaBuffer::allocate(width, height, format) {
        Ok(mut dma_buf) => {
            assert!(dma_buf.size() >= (width * height * 4) as usize);
            assert!(dma_buf.stride() >= (width * 4) as usize);
            assert!(!dma_buf.host_ptr().is_null());

            // Test scoped with_write closure
            let write_res = dma_buf.with_write(|slice| {
                slice[0] = 0xDE;
                slice[1] = 0xAD;
                slice.len()
            });
            assert!(write_res.is_ok());

            // Test scoped with_read closure
            let read_val = dma_buf.with_read(|slice| (slice[0], slice[1]));
            assert_eq!(read_val.unwrap(), (0xDE, 0xAD));

            // Verify ImageDesc views
            let desc = dma_buf.as_image_desc();
            assert_eq!(desc.width, width);
            assert_eq!(desc.height, height);
            assert_eq!(desc.data[0], 0xDE);
            assert_eq!(desc.data[1], 0xAD);
            assert!(desc.dma_buf_fd.is_some());
            assert_eq!(dma_buf.dimensions(), ImageDimensions::new(width, height));
            assert_eq!(dma_buf.width(), width);
            assert_eq!(dma_buf.height(), height);
        }
        Err(ScalixError::DmaUnavailable(reason)) => {
            // Expected on virtualized/container environments without DMA-Heap/DRM hardware nodes
            println!(
                "DMA allocator safely probed as unavailable on this host: {}",
                reason
            );
        }
        Err(other) => {
            panic!("Unexpected DMA error: {:?}", other);
        }
    }

    // Dimension validation test
    assert!(DmaBuffer::allocate_dimensions(ImageDimensions::new(0, 64), format).is_err());
    assert!(DmaBuffer::allocate_dimensions(ImageDimensions::new(64, 0), format).is_err());
}

#[test]
fn test_vulkan_backend_blit_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!(
                "Vulkan backend not available on this environment ({:?}); skipping test.",
                e
            );
            return;
        }
    };

    // 2. Perform 64x64 -> 32x32 downscale with dynamic Blit strategy
    let src_w = 64;
    let src_h = 64;
    let dst_w = 32;
    let dst_h = 32;
    let format = PixelFormat::Rgba8888;

    let src_stride = format.min_stride(src_w).unwrap();
    let dst_stride = format.min_stride(dst_w).unwrap();

    let mut src_data = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
    src_data.as_mut_slice().fill(0xAA);
    let mut dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    let options =
        ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Blit);
    let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
    assert!(
        process_res.is_ok(),
        "Vulkan blit process failed: {:?}",
        process_res.err()
    );

    // Verify destination pixels were populated by Vulkan GPU blit
    assert_eq!(dst_data.as_slice()[0], 0xAA);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0xAA);
}

#[test]
fn test_vulkan_backend_raster_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!(
                "Vulkan backend not available on this environment ({:?}); skipping test.",
                e
            );
            return;
        }
    };

    // 2. Perform 64x64 -> 32x32 downscale with Bilinear sampling and dynamic Raster strategy
    let src_w = 64;
    let src_h = 64;
    let dst_w = 32;
    let dst_h = 32;
    let format = PixelFormat::Rgba8888;

    let src_stride = format.min_stride(src_w).unwrap();
    let dst_stride = format.min_stride(dst_w).unwrap();

    let mut src_data = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
    src_data.as_mut_slice().fill(0xBB);
    let mut dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    let options =
        ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Raster);
    let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
    assert!(
        process_res.is_ok(),
        "Vulkan raster process failed: {:?}",
        process_res.err()
    );

    // Verify destination pixels were populated by Vulkan GPU rasterization
    assert_eq!(dst_data.as_slice()[0], 0xBB);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0xBB);

    // 3. Test Packed RGB888 format with dynamic Raster strategy
    let rgb_format = PixelFormat::Rgb888;
    let rgb_src_stride = rgb_format.min_stride(src_w).unwrap();
    let rgb_dst_stride = rgb_format.min_stride(dst_w).unwrap();
    let mut rgb_src_data = AlignedBuffer::new(rgb_src_stride * (src_h as usize)).unwrap();
    rgb_src_data.as_mut_slice().fill(0xCC);
    let mut rgb_dst_data = AlignedBuffer::new(rgb_dst_stride * (dst_h as usize)).unwrap();

    let rgb_src_desc =
        ImageDesc::new(src_w, src_h, rgb_src_stride, rgb_format, &rgb_src_data).unwrap();
    let mut rgb_dst_desc =
        ImageDescMut::new(dst_w, dst_h, rgb_dst_stride, rgb_format, &mut rgb_dst_data).unwrap();

    let rgb_options =
        ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Raster);
    let rgb_res = vk_backend.process(&rgb_src_desc, &mut rgb_dst_desc, &rgb_options);
    assert!(
        rgb_res.is_ok(),
        "Vulkan raster RGB888 process failed: {:?}",
        rgb_res.err()
    );
    assert_eq!(rgb_dst_data.as_slice()[0], 0xCC);
    assert_eq!(rgb_dst_data.as_slice()[rgb_dst_data.len() - 1], 0xCC);
}

#[test]
fn test_vulkan_backend_lod_pyramid_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!(
                "Vulkan backend not available on this environment ({:?}); skipping test.",
                e
            );
            return;
        }
    };

    // 2. Perform 128x128 -> 16x16 downscale (8x reduction across 3 mip levels)
    let src_w = 128;
    let src_h = 128;
    let dst_w = 16;
    let dst_h = 16;
    let format = PixelFormat::Rgba8888;

    let src_stride = format.min_stride(src_w).unwrap();
    let dst_stride = format.min_stride(dst_w).unwrap();

    let mut src_data = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
    src_data.as_mut_slice().fill(0xDD);
    let mut dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    {
        let mut dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options = ResizeOptions::new(FilterMode::Bilinear)
            .with_vulkan_strategy(VulkanStrategy::LodPyramid);
        let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(
            process_res.is_ok(),
            "Vulkan LoD pyramid process failed: {:?}",
            process_res.err()
        );
    }

    // Verify destination pixels were populated by Vulkan GPU LoD downscaler
    assert_eq!(dst_data.as_slice()[0], 0xDD);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0xDD);

    // 3. Test Packed RGB888 format with LodPyramid strategy
    let rgb_format = PixelFormat::Rgb888;
    let rgb_src_stride = rgb_format.min_stride(src_w).unwrap();
    let rgb_dst_stride = rgb_format.min_stride(dst_w).unwrap();
    let mut rgb_src_data = AlignedBuffer::new(rgb_src_stride * (src_h as usize)).unwrap();
    rgb_src_data.as_mut_slice().fill(0xEE);
    let mut rgb_dst_data = AlignedBuffer::new(rgb_dst_stride * (dst_h as usize)).unwrap();

    let rgb_src_desc =
        ImageDesc::new(src_w, src_h, rgb_src_stride, rgb_format, &rgb_src_data).unwrap();
    {
        let mut rgb_dst_desc =
            ImageDescMut::new(dst_w, dst_h, rgb_dst_stride, rgb_format, &mut rgb_dst_data).unwrap();
        let rgb_options = ResizeOptions::new(FilterMode::Bilinear)
            .with_vulkan_strategy(VulkanStrategy::LodPyramid);
        let rgb_res = vk_backend.process(&rgb_src_desc, &mut rgb_dst_desc, &rgb_options);
        assert!(
            rgb_res.is_ok(),
            "Vulkan LoD RGB888 process failed: {:?}",
            rgb_res.err()
        );
    }
    assert_eq!(rgb_dst_data.as_slice()[0], 0xEE);
    assert_eq!(rgb_dst_data.as_slice()[rgb_dst_data.len() - 1], 0xEE);

    // 4. Test with explicit max_mip_levels limit (capped at 2 levels)
    let mut capped_dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();
    {
        let mut capped_dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut capped_dst_data).unwrap();
        let capped_options = ResizeOptions::new(FilterMode::Bilinear)
            .with_vulkan_strategy(VulkanStrategy::LodPyramid)
            .with_max_mip_levels(2);
        let capped_res = vk_backend.process(&src_desc, &mut capped_dst_desc, &capped_options);
        assert!(
            capped_res.is_ok(),
            "Vulkan LoD capped process failed: {:?}",
            capped_res.err()
        );
    }
    assert_eq!(capped_dst_data.as_slice()[0], 0xDD);
}

#[test]
fn test_pluggable_profiler_and_gpu_metrics() {
    let engine = match Engine::new(BackendType::Vulkan) {
        Ok(e) => e,
        Err(e) => {
            println!("Vulkan backend unavailable ({:?}); skipping test.", e);
            return;
        }
    };

    // 1. Profiler disabled by default -> last_profile should be None
    assert_eq!(engine.last_profile(), None);

    let src_w = 64;
    let src_h = 64;
    let dst_w = 32;
    let dst_h = 32;
    let format = PixelFormat::Rgb888;

    let src_stride = format.min_stride(src_w).unwrap();
    let dst_stride = format.min_stride(dst_w).unwrap();

    let mut src_data = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
    src_data.as_mut_slice().fill(128);
    let mut dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    engine
        .resize_sync(&src_desc, &mut dst_desc, FilterMode::Bilinear)
        .unwrap();
    assert_eq!(engine.last_profile(), None);

    // 2. Enable profiling
    engine.set_profiling(true);
    engine
        .resize_sync(&src_desc, &mut dst_desc, FilterMode::Bilinear)
        .unwrap();

    let profile = engine
        .last_profile()
        .expect("Expected profile metrics when enabled");
    assert!(profile.total_wall_ms > 0.0);
    assert!(profile.gpu_pure_blit_ms >= 0.0);
}

#[test]
fn test_image_dimensions_and_geometry() {
    let dims = ImageDimensions::new(64, 48);
    assert!(!dims.is_empty());
    assert_eq!(dims.checked_area(), Some(64 * 48));
    assert!(dims.validate_even().is_ok());

    let empty_dims = ImageDimensions::new(0, 100);
    assert!(empty_dims.is_empty());

    let odd_dims = ImageDimensions::new(63, 48);
    assert!(matches!(
        odd_dims.validate_even(),
        Err(ScalixError::InvalidDimensions { .. })
    ));

    let geom = ImageGeometry::new(dims, PixelFormat::Rgba8888).unwrap();
    assert_eq!(geom.dimensions, dims);
    assert_eq!(geom.stride, 64 * 4);
    assert_eq!(geom.min_buffer_size().unwrap(), 64 * 48 * 4);

    let owned = OwnedImage::allocate_geometry(geom).unwrap();
    assert_eq!(owned.dimensions(), dims);
    assert_eq!(owned.stride(), 64 * 4);
    assert_eq!(owned.format(), PixelFormat::Rgba8888);
    assert_eq!(owned.geometry(), geom);

    let desc = ImageDesc::from_geometry(geom, owned.data()).unwrap();
    assert_eq!(desc.dimensions(), dims);
    assert_eq!(desc.stride, geom.stride);

    let mut owned_mut = owned.clone();
    let desc_mut = ImageDescMut::from_geometry(geom, owned_mut.data_mut()).unwrap();
    assert_eq!(desc_mut.dimensions(), dims);
}

#[test]
fn test_engine_config_and_worker_pool() {
    let config = EngineConfig::new()
        .with_thread_prefix("cfg_eng")
        .with_hw_threads(1)
        .with_callback_workers(2)
        .with_queue_capacity(512);

    let backend = Arc::new(scalix_core::PassthroughBackend::new());
    let engine = Engine::with_config(backend, config);

    assert!(engine.backend_name().contains("Passthrough"));
    assert_eq!(engine.backend_type(), BackendType::Passthrough);

    let dims = ImageDimensions::new(16, 16);
    let geom = ImageGeometry::new(dims, PixelFormat::Rgba8888).unwrap();
    let mut src = OwnedImage::allocate_geometry(geom).unwrap();
    src.data_mut().fill(0x7F);
    let dst = OwnedImage::allocate_geometry(geom).unwrap();

    let task = engine.resize_async(src, dst, FilterMode::Passthrough);
    let completed = task.wait(Some(Duration::from_secs(2))).unwrap();
    assert_eq!(completed.data()[0], 0x7F);

    // Test WorkerPool try_with_prefix directly
    let pool = WorkerPool::try_with_prefix("tst-p", 2, 256).unwrap();
    let (tx, rx) = crossbeam_channel::bounded(1);
    pool.submit(move || {
        let _ = tx.send(42);
    })
    .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), 42);
}

#[test]
fn test_custom_profiler_default_implementations() {
    struct MinimalProfiler;
    impl Profiler for MinimalProfiler {
        fn is_enabled(&self) -> bool {
            false
        }
        fn record(&self, _metrics: ProfileMetrics) {}
    }

    let p = MinimalProfiler;
    assert!(!p.is_enabled());
    assert_eq!(p.last_metrics(), None);
    p.set_enabled(true); // Should not panic
}

#[test]
fn test_vulkan_pipeline_trait_and_dma_allocator() {
    use scalix_core::backend::vulkan::VulkanPipeline;
    use scalix_core::dma::DmaAllocator;

    #[cfg(target_os = "linux")]
    {
        let heap_alloc = scalix_core::dma::linux_dma_heap::LinuxDmaHeapAllocator;
        assert_eq!(heap_alloc.name(), "linux_dma_heap");

        let drm_alloc = scalix_core::dma::linux_drm::LinuxDrmAllocator;
        assert_eq!(drm_alloc.name(), "linux_drm");
    }

    struct MockPipeline;
    impl VulkanPipeline for MockPipeline {
        fn name(&self) -> &'static str {
            "MockPipeline"
        }
        fn strategy(&self) -> scalix_core::VulkanStrategy {
            scalix_core::VulkanStrategy::Blit
        }
        fn process(
            &self,
            _src: &scalix_core::ImageDesc,
            _dst: &mut scalix_core::ImageDescMut,
            _options: &scalix_core::ResizeOptions,
        ) -> scalix_core::Result<()> {
            Ok(())
        }
    }

    let pipeline: Box<dyn VulkanPipeline> = Box::new(MockPipeline);
    assert_eq!(pipeline.name(), "MockPipeline");
    assert_eq!(pipeline.strategy(), scalix_core::VulkanStrategy::Blit);
}

#[test]
fn test_cpu_rgb888_unpack_and_repack_4byte_masked() {
    use scalix_core::backend::vulkan::{cpu_repack_rgb888, cpu_unpack_rgb888};

    for num_pixels in [1, 2, 3, 4, 5, 7, 8, 15, 16, 33, 1024] {
        let mut original_rgb = AlignedBuffer::new(num_pixels * 3).unwrap();
        {
            let rgb_slice = original_rgb.as_mut_slice();
            for p in 0..num_pixels {
                rgb_slice[p * 3] = (p * 7 % 256) as u8;
                rgb_slice[p * 3 + 1] = (p * 13 % 256) as u8;
                rgb_slice[p * 3 + 2] = (p * 29 % 256) as u8;
            }
        }

        let mut rgba = AlignedBuffer::new(num_pixels * 4).unwrap();
        cpu_unpack_rgb888(original_rgb.as_slice(), rgba.as_mut_ptr(), num_pixels);

        let rgba_slice = rgba.as_slice();
        let rgb_slice = original_rgb.as_slice();
        // Verify unpacked RGBA channels
        for p in 0..num_pixels {
            assert_eq!(
                rgba_slice[p * 4],
                rgb_slice[p * 3],
                "R mismatch at pixel {p}"
            );
            assert_eq!(
                rgba_slice[p * 4 + 1],
                rgb_slice[p * 3 + 1],
                "G mismatch at pixel {p}"
            );
            assert_eq!(
                rgba_slice[p * 4 + 2],
                rgb_slice[p * 3 + 2],
                "B mismatch at pixel {p}"
            );
            assert_eq!(rgba_slice[p * 4 + 3], 255, "Alpha mismatch at pixel {p}");
        }

        // Repack back to RGB
        let mut repacked_rgb = AlignedBuffer::new(num_pixels * 3).unwrap();
        cpu_repack_rgb888(rgba.as_ptr(), repacked_rgb.as_mut_slice(), num_pixels);

        assert_eq!(
            repacked_rgb.as_slice(),
            original_rgb.as_slice(),
            "Roundtrip mismatch for {num_pixels} pixels"
        );
    }
}

#[test]
fn test_vulkan_backend_compute_direct_rgb888_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!(
                "Vulkan backend not available on this environment ({:?}); skipping test.",
                e
            );
            return;
        }
    };

    let src_w = 64;
    let src_h = 64;
    let dst_w = 32;
    let dst_h = 32;
    let format = PixelFormat::Rgb888;

    let src_stride = format.min_stride(src_w).unwrap();
    let dst_stride = format.min_stride(dst_w).unwrap();

    let mut src_data = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
    src_data.as_mut_slice().fill(0x77);
    let mut dst_data = AlignedBuffer::new(dst_stride * (dst_h as usize)).unwrap();

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();

    // 1. Test Nearest Filter via Direct Compute
    {
        let mut dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options =
            ResizeOptions::new(FilterMode::Nearest).with_vulkan_strategy(VulkanStrategy::Compute);
        let res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(res.is_ok(), "Compute Nearest failed: {:?}", res.err());
    }
    assert_eq!(dst_data.as_slice()[0], 0x77);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0x77);

    // 2. Test Bilinear Filter via Direct Compute
    {
        let mut dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options =
            ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Compute);
        let res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(res.is_ok(), "Compute Bilinear failed: {:?}", res.err());
    }
    assert_eq!(dst_data.as_slice()[0], 0x77);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0x77);

    // 3. Test Auto Strategy routing directly to Compute for RGB888
    {
        let mut dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options =
            ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Auto);
        let res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(
            res.is_ok(),
            "Auto Strategy (Compute) failed: {:?}",
            res.err()
        );
    }
    assert_eq!(dst_data.as_slice()[0], 0x77);
    assert_eq!(dst_data.as_slice()[dst_data.len() - 1], 0x77);

    // 4. Test Dynamic Pluggable Shader Registration (e.g. registering custom kernel for Lanczos3)
    {
        let pluggable_spv = scalix_core::backend::vulkan::compute::RGB888_RESIZE_NEAREST_COMP_SPV;
        assert!(!vk_backend
            .compute_resizer()
            .has_shader(FilterMode::Lanczos3));

        let reg_res = vk_backend.register_compute_shader(FilterMode::Lanczos3, pluggable_spv);
        assert!(
            reg_res.is_ok(),
            "Failed to register custom compute shader: {:?}",
            reg_res.err()
        );
        assert!(vk_backend
            .compute_resizer()
            .has_shader(FilterMode::Lanczos3));

        let mut dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options =
            ResizeOptions::new(FilterMode::Lanczos3).with_vulkan_strategy(VulkanStrategy::Compute);
        let res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(
            res.is_ok(),
            "Pluggable custom shader execution failed: {:?}",
            res.err()
        );
        assert_eq!(dst_data.as_slice()[0], 0x77);
    }

    // 5. Test Non-Uniform Pattern to verify channel ordering and spatial scaling
    {
        let mut patterned_src = AlignedBuffer::new(src_stride * (src_h as usize)).unwrap();
        for y in 0..src_h {
            for x in 0..src_w {
                let idx = (y as usize) * src_stride + (x as usize) * 3;
                patterned_src.as_mut_slice()[idx] = (x * 4) as u8;
                patterned_src.as_mut_slice()[idx + 1] = (y * 4) as u8;
                patterned_src.as_mut_slice()[idx + 2] = 0xAA;
            }
        }
        let pattern_src_desc =
            ImageDesc::new(src_w, src_h, src_stride, format, &patterned_src).unwrap();
        let mut pattern_dst_desc =
            ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options =
            ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Compute);
        let res = vk_backend.process(&pattern_src_desc, &mut pattern_dst_desc, &options);
        assert!(
            res.is_ok(),
            "Compute Bilinear patterned failed: {:?}",
            res.err()
        );
        assert_eq!(dst_data.as_slice()[2], 0xAA, "Blue channel mismatch");
    }
}
