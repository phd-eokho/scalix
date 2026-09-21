use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use scalix_core::{
    BackendType, Engine, FilterMode, ImageDesc, ImageDescMut, OwnedImage, PixelFormat, ScalixError,
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

