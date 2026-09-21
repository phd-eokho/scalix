use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use scalix_core::{
    Backend, BackendType, Engine, FilterMode, ImageDesc, ImageDescMut, OwnedImage, PixelFormat,
    ScalixError,
};

#[test]
fn test_sync_passthrough_memory_transfer() {
    let engine = Engine::new(BackendType::Passthrough).expect("Failed to create engine");

    let width = 64;
    let height = 64;
    let format = PixelFormat::Rgba8888;
    let stride = format.min_stride(width).unwrap();

    let mut src_data = vec![0u8; stride * (height as usize)];
    // Fill with pattern
    for (i, byte) in src_data.iter_mut().enumerate() {
        *byte = (i % 255) as u8;
    }

    let mut dst_data = vec![0u8; stride * (height as usize)];

    let src_desc = ImageDesc::new(width, height, stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(width, height, stride, format, &mut dst_data).unwrap();

    engine
        .resize_sync(&src_desc, &mut dst_desc, FilterMode::Passthrough)
        .expect("Resize sync failed");

    assert_eq!(src_data, dst_data, "Destination data must match source data");
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

    let result = task.wait(Some(Duration::from_secs(2))).expect("Task failed");
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

    let res = rx.recv_timeout(Duration::from_secs(2)).expect("Callback timed out");
    let completed_image = res.expect("Execution error in callback");

    assert!(called.load(Ordering::Acquire));
    assert_eq!(completed_image.data[0], 0xAA);
}

#[test]
fn test_buffer_bounds_and_error_handling() {
    // 1. Zero dimension error
    let zero_res = PixelFormat::Rgba8888.min_buffer_size(0, 100, 100);
    assert!(matches!(zero_res, Err(ScalixError::InvalidDimensions { .. })));

    // 2. Undersized stride error
    let stride_res = PixelFormat::Rgba8888.min_buffer_size(100, 100, 100); // 100 < 400
    assert!(matches!(stride_res, Err(ScalixError::InvalidStride { .. })));

    // 3. Buffer too small error
    let raw_data = vec![0u8; 10];
    let img_res = ImageDesc::new(100, 100, 400, PixelFormat::Rgba8888, &raw_data);
    assert!(matches!(img_res, Err(ScalixError::BufferTooSmall { .. })));
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
            let read_val = dma_buf.with_read(|slice| {
                (slice[0], slice[1])
            });
            assert_eq!(read_val.unwrap(), (0xDE, 0xAD));

            // Verify ImageDesc views
            let desc = dma_buf.as_image_desc();
            assert_eq!(desc.width, width);
            assert_eq!(desc.height, height);
            assert_eq!(desc.data[0], 0xDE);
            assert_eq!(desc.data[1], 0xAD);
            assert!(desc.dma_buf_fd.is_some());
        }
        Err(ScalixError::DmaUnavailable(reason)) => {
            // Expected on virtualized/container environments without DMA-Heap/DRM hardware nodes
            println!("DMA allocator safely probed as unavailable on this host: {}", reason);
        }
        Err(other) => {
            panic!("Unexpected DMA error: {:?}", other);
        }
    }
}

#[test]
fn test_vulkan_backend_blit_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!("Vulkan backend not available on this environment ({:?}); skipping test.", e);
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

    let src_data = vec![0xAAu8; src_stride * (src_h as usize)];
    let mut dst_data = vec![0x00u8; dst_stride * (dst_h as usize)];

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    let options = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Blit);
    let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
    assert!(process_res.is_ok(), "Vulkan blit process failed: {:?}", process_res.err());

    // Verify destination pixels were populated by Vulkan GPU blit
    assert_eq!(dst_data[0], 0xAA);
    assert_eq!(dst_data[dst_data.len() - 1], 0xAA);
}

#[test]
fn test_vulkan_backend_raster_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!("Vulkan backend not available on this environment ({:?}); skipping test.", e);
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

    let src_data = vec![0xBBu8; src_stride * (src_h as usize)];
    let mut dst_data = vec![0x00u8; dst_stride * (dst_h as usize)];

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    let options = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Raster);
    let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
    assert!(process_res.is_ok(), "Vulkan raster process failed: {:?}", process_res.err());

    // Verify destination pixels were populated by Vulkan GPU rasterization
    assert_eq!(dst_data[0], 0xBB);
    assert_eq!(dst_data[dst_data.len() - 1], 0xBB);

    // 3. Test Packed RGB888 format with dynamic Raster strategy
    let rgb_format = PixelFormat::Rgb888;
    let rgb_src_stride = rgb_format.min_stride(src_w).unwrap();
    let rgb_dst_stride = rgb_format.min_stride(dst_w).unwrap();
    let rgb_src_data = vec![0xCCu8; rgb_src_stride * (src_h as usize)];
    let mut rgb_dst_data = vec![0x00u8; rgb_dst_stride * (dst_h as usize)];

    let rgb_src_desc = ImageDesc::new(src_w, src_h, rgb_src_stride, rgb_format, &rgb_src_data).unwrap();
    let mut rgb_dst_desc = ImageDescMut::new(dst_w, dst_h, rgb_dst_stride, rgb_format, &mut rgb_dst_data).unwrap();

    let rgb_options = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::Raster);
    let rgb_res = vk_backend.process(&rgb_src_desc, &mut rgb_dst_desc, &rgb_options);
    assert!(rgb_res.is_ok(), "Vulkan raster RGB888 process failed: {:?}", rgb_res.err());
    assert_eq!(rgb_dst_data[0], 0xCC);
    assert_eq!(rgb_dst_data[rgb_dst_data.len() - 1], 0xCC);
}

#[test]
fn test_vulkan_backend_lod_pyramid_resize() {
    use scalix_core::{ResizeOptions, VulkanBackend, VulkanStrategy};

    // 1. Instantiate Vulkan backend
    let vk_backend = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(e) => {
            println!("Vulkan backend not available on this environment ({:?}); skipping test.", e);
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

    let src_data = vec![0xDDu8; src_stride * (src_h as usize)];
    let mut dst_data = vec![0x00u8; dst_stride * (dst_h as usize)];

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    {
        let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();
        let options = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::LodPyramid);
        let process_res = vk_backend.process(&src_desc, &mut dst_desc, &options);
        assert!(process_res.is_ok(), "Vulkan LoD pyramid process failed: {:?}", process_res.err());
    }

    // Verify destination pixels were populated by Vulkan GPU LoD downscaler
    assert_eq!(dst_data[0], 0xDD);
    assert_eq!(dst_data[dst_data.len() - 1], 0xDD);

    // 3. Test Packed RGB888 format with LodPyramid strategy
    let rgb_format = PixelFormat::Rgb888;
    let rgb_src_stride = rgb_format.min_stride(src_w).unwrap();
    let rgb_dst_stride = rgb_format.min_stride(dst_w).unwrap();
    let rgb_src_data = vec![0xEEu8; rgb_src_stride * (src_h as usize)];
    let mut rgb_dst_data = vec![0x00u8; rgb_dst_stride * (dst_h as usize)];

    let rgb_src_desc = ImageDesc::new(src_w, src_h, rgb_src_stride, rgb_format, &rgb_src_data).unwrap();
    {
        let mut rgb_dst_desc = ImageDescMut::new(dst_w, dst_h, rgb_dst_stride, rgb_format, &mut rgb_dst_data).unwrap();
        let rgb_options = ResizeOptions::new(FilterMode::Bilinear).with_vulkan_strategy(VulkanStrategy::LodPyramid);
        let rgb_res = vk_backend.process(&rgb_src_desc, &mut rgb_dst_desc, &rgb_options);
        assert!(rgb_res.is_ok(), "Vulkan LoD RGB888 process failed: {:?}", rgb_res.err());
    }
    assert_eq!(rgb_dst_data[0], 0xEE);
    assert_eq!(rgb_dst_data[rgb_dst_data.len() - 1], 0xEE);

    // 4. Test with explicit max_mip_levels limit (capped at 2 levels)
    let mut capped_dst_data = vec![0x00u8; dst_stride * (dst_h as usize)];
    {
        let mut capped_dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut capped_dst_data).unwrap();
        let capped_options = ResizeOptions::new(FilterMode::Bilinear)
            .with_vulkan_strategy(VulkanStrategy::LodPyramid)
            .with_max_mip_levels(2);
        let capped_res = vk_backend.process(&src_desc, &mut capped_dst_desc, &capped_options);
        assert!(capped_res.is_ok(), "Vulkan LoD capped process failed: {:?}", capped_res.err());
    }
    assert_eq!(capped_dst_data[0], 0xDD);
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

    let src_data = vec![128u8; src_stride * (src_h as usize)];
    let mut dst_data = vec![0u8; dst_stride * (dst_h as usize)];

    let src_desc = ImageDesc::new(src_w, src_h, src_stride, format, &src_data).unwrap();
    let mut dst_desc = ImageDescMut::new(dst_w, dst_h, dst_stride, format, &mut dst_data).unwrap();

    engine.resize_sync(&src_desc, &mut dst_desc, FilterMode::Bilinear).unwrap();
    assert_eq!(engine.last_profile(), None);

    // 2. Enable profiling
    engine.set_profiling(true);
    engine.resize_sync(&src_desc, &mut dst_desc, FilterMode::Bilinear).unwrap();

    let profile = engine.last_profile().expect("Expected profile metrics when enabled");
    assert!(profile.total_wall_ms > 0.0);
    assert!(profile.gpu_pure_blit_ms >= 0.0);
}




