use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use scalix::*;

#[test]
fn test_c_api_sync_passthrough() {
    unsafe {
        let engine = scalix_engine_create(ScalixBackendType::Auto);
        assert!(!engine.is_null());

        let width = 16;
        let height = 16;
        let stride = width * 4;
        let mut src_data = scalix_core::AlignedBuffer::new(stride * height).unwrap();
        src_data.as_mut_slice().fill(42);
        let mut dst_data = scalix_core::AlignedBuffer::new(stride * height).unwrap();

        let src_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: src_data.as_mut_ptr(),
            data_len: src_data.len(),
            dma_buf_fd: -1,
        };

        let mut dst_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: dst_data.as_mut_ptr(),
            data_len: dst_data.len(),
            dma_buf_fd: -1,
        };

        let status = scalix_resize_sync(
            engine,
            &src_desc,
            &mut dst_desc,
            ScalixFilterMode::Passthrough,
        );
        assert_eq!(status, SCALIX_SUCCESS);
        assert_eq!(dst_data.as_slice(), src_data.as_slice());

        scalix_engine_destroy(engine);
    }
}

#[test]
fn test_c_api_async_wait() {
    unsafe {
        let engine = scalix_engine_create(ScalixBackendType::Auto);
        assert!(!engine.is_null());

        let width = 8;
        let height = 8;
        let stride = width * 4;
        let mut src_data = scalix_core::AlignedBuffer::new(stride * height).unwrap();
        src_data.as_mut_slice().fill(99);
        let mut dst_data = scalix_core::AlignedBuffer::new(stride * height).unwrap();

        let src_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: src_data.as_mut_ptr(),
            data_len: src_data.len(),
            dma_buf_fd: -1,
        };

        let dst_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: std::ptr::null_mut(),
            data_len: 0,
            dma_buf_fd: -1,
        };

        let task = scalix_resize_async(engine, &src_desc, &dst_desc, ScalixFilterMode::Passthrough);
        assert!(!task.is_null());

        let status = scalix_task_wait(task, 1000, dst_data.as_mut_ptr(), dst_data.len());
        assert_eq!(status, SCALIX_SUCCESS);
        assert_eq!(dst_data.as_slice(), src_data.as_slice());

        scalix_task_release(task);
        scalix_engine_destroy(engine);
    }
}

unsafe extern "C" fn test_callback_fn(status_code: i32, user_data: *mut c_void) {
    let flag = &*(user_data as *const AtomicBool);
    if status_code == SCALIX_SUCCESS {
        flag.store(true, Ordering::Release);
    }
}

#[test]
fn test_c_api_callback_submit() {
    unsafe {
        let engine = scalix_engine_create(ScalixBackendType::Auto);
        assert!(!engine.is_null());

        let width = 8;
        let height = 8;
        let stride = width * 4;
        let mut src_data = scalix_core::AlignedBuffer::new(stride * height).unwrap();
        src_data.as_mut_slice().fill(77);

        let src_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: src_data.as_mut_ptr(),
            data_len: src_data.len(),
            dma_buf_fd: -1,
        };

        let dst_desc = ScalixImageDesc {
            width: width as u32,
            height: height as u32,
            stride_bytes: stride,
            format: ScalixPixelFormat::Rgba8888,
            host_ptr: std::ptr::null_mut(),
            data_len: 0,
            dma_buf_fd: -1,
        };

        let flag = Arc::new(AtomicBool::new(false));
        let user_data = Arc::as_ptr(&flag) as *mut c_void;

        let status = scalix_resize_submit(
            engine,
            &src_desc,
            &dst_desc,
            ScalixFilterMode::Passthrough,
            Some(test_callback_fn),
            user_data,
        );
        assert_eq!(status, SCALIX_SUCCESS);

        // Poll for callback completion
        for _ in 0..100 {
            if flag.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(flag.load(Ordering::Acquire));

        scalix_engine_destroy(engine);
    }
}

#[test]
fn test_c_api_with_custom_prefix() {
    use std::ffi::CString;
    unsafe {
        let prefix = CString::new("testapp").unwrap();
        let engine = scalix_engine_create_with_prefix(ScalixBackendType::Auto, prefix.as_ptr());
        assert!(!engine.is_null());
        scalix_engine_destroy(engine);
    }
}

#[test]
fn test_c_api_dma_buffer() {
    unsafe {
        let dma_buf = scalix_dma_buffer_allocate(32, 32, ScalixPixelFormat::Rgba8888);
        if !dma_buf.is_null() {
            let host_ptr = scalix_dma_buffer_get_host_ptr(dma_buf);
            assert!(!host_ptr.is_null());
            assert!(scalix_dma_buffer_get_size(dma_buf) >= 32 * 32 * 4);
            assert_eq!(scalix_dma_buffer_sync_start(dma_buf, true), SCALIX_SUCCESS);
            assert_eq!(scalix_dma_buffer_sync_end(dma_buf, true), SCALIX_SUCCESS);

            let mut desc = ScalixImageDesc {
                width: 0,
                height: 0,
                stride_bytes: 0,
                format: ScalixPixelFormat::Rgba8888,
                host_ptr: std::ptr::null_mut(),
                data_len: 0,
                dma_buf_fd: -1,
            };
            assert_eq!(
                scalix_dma_buffer_get_desc(dma_buf, &mut desc),
                SCALIX_SUCCESS
            );
            assert_eq!(desc.width, 32);
            assert_eq!(desc.height, 32);

            scalix_dma_buffer_free(dma_buf);
        }
    }
}

#[test]
fn test_c_api_profiling() {
    unsafe {
        let engine = scalix_engine_create(ScalixBackendType::Auto);
        assert!(!engine.is_null());

        let mut metrics = ScalixProfileMetrics::default();
        let res = scalix_engine_get_last_profile(engine, &mut metrics);
        assert_ne!(res, SCALIX_SUCCESS); // Inactive -> error / None

        assert_eq!(scalix_engine_set_profiling(engine, true), SCALIX_SUCCESS);

        scalix_engine_destroy(engine);
    }
}
