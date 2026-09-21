use std::ffi::c_void;
use std::slice;
use std::time::Duration;

use scalix_core::{
    BackendType, Engine, FilterMode, ImageDesc, ImageDescMut, OwnedImage, PixelFormat, ScalixError,
    TaskHandle,
};

pub const SCALIX_SUCCESS: i32 = 0;
pub const SCALIX_ERR_NULL_PTR: i32 = -1;
pub const SCALIX_ERR_INVALID_DIMENSIONS: i32 = -2;
pub const SCALIX_ERR_INVALID_STRIDE: i32 = -3;
pub const SCALIX_ERR_BUFFER_TOO_SMALL: i32 = -4;
pub const SCALIX_ERR_UNSUPPORTED_FORMAT: i32 = -5;
pub const SCALIX_ERR_BACKEND_UNAVAILABLE: i32 = -6;
pub const SCALIX_ERR_TIMEOUT: i32 = -7;
pub const SCALIX_ERR_DMA_UNAVAILABLE: i32 = -8;
pub const SCALIX_ERR_DMA_ALLOCATION_FAILED: i32 = -9;
pub const SCALIX_ERR_DMA_MAP_FAILED: i32 = -10;
pub const SCALIX_ERR_DMA_SYNC_FAILED: i32 = -11;
pub const SCALIX_ERR_FAILED: i32 = -99;

fn map_error_to_code(err: ScalixError) -> i32 {
    match err {
        ScalixError::NullPointer => SCALIX_ERR_NULL_PTR,
        ScalixError::InvalidDimensions { .. } => SCALIX_ERR_INVALID_DIMENSIONS,
        ScalixError::InvalidStride { .. } => SCALIX_ERR_INVALID_STRIDE,
        ScalixError::BufferTooSmall { .. } => SCALIX_ERR_BUFFER_TOO_SMALL,
        ScalixError::UnsupportedFormat(_) => SCALIX_ERR_UNSUPPORTED_FORMAT,
        ScalixError::BackendUnavailable(_) => SCALIX_ERR_BACKEND_UNAVAILABLE,
        ScalixError::Timeout => SCALIX_ERR_TIMEOUT,
        ScalixError::DmaUnavailable(_) => SCALIX_ERR_DMA_UNAVAILABLE,
        ScalixError::DmaAllocationFailed(_) => SCALIX_ERR_DMA_ALLOCATION_FAILED,
        ScalixError::DmaMapFailed(_) => SCALIX_ERR_DMA_MAP_FAILED,
        ScalixError::DmaSyncFailed(_) => SCALIX_ERR_DMA_SYNC_FAILED,
        _ => SCALIX_ERR_FAILED,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalixBackendType {
    Auto = 0,
    Vulkan = 1,
    OpenGL = 2,
    Npu = 3,
    Hw2d = 4,
    Cpu = 5,
    Passthrough = 6,
}

impl From<ScalixBackendType> for BackendType {
    fn from(b: ScalixBackendType) -> Self {
        match b {
            ScalixBackendType::Auto => BackendType::Auto,
            ScalixBackendType::Vulkan => BackendType::Vulkan,
            ScalixBackendType::OpenGL => BackendType::OpenGL,
            ScalixBackendType::Npu => BackendType::Npu,
            ScalixBackendType::Hw2d => BackendType::Hw2d,
            ScalixBackendType::Cpu => BackendType::Cpu,
            ScalixBackendType::Passthrough => BackendType::Passthrough,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalixFilterMode {
    Nearest = 0,
    Bilinear = 1,
    Bicubic = 2,
    Lanczos3 = 3,
    Area = 4,
    Passthrough = 100,
}

impl From<ScalixFilterMode> for FilterMode {
    fn from(f: ScalixFilterMode) -> Self {
        match f {
            ScalixFilterMode::Nearest => FilterMode::Nearest,
            ScalixFilterMode::Bilinear => FilterMode::Bilinear,
            ScalixFilterMode::Bicubic => FilterMode::Bicubic,
            ScalixFilterMode::Lanczos3 => FilterMode::Lanczos3,
            ScalixFilterMode::Area => FilterMode::Area,
            ScalixFilterMode::Passthrough => FilterMode::Passthrough,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalixPixelFormat {
    Rgba8888 = 0,
    Bgra8888 = 1,
    Rgb888 = 2,
    Bgr888 = 3,
    R8 = 4,
    Rg88 = 5,
    Nv12 = 6,
    Yuv420p = 7,
    Rgba16f = 8,
    Rgba32f = 9,
}

impl From<ScalixPixelFormat> for PixelFormat {
    fn from(f: ScalixPixelFormat) -> Self {
        match f {
            ScalixPixelFormat::Rgba8888 => PixelFormat::Rgba8888,
            ScalixPixelFormat::Bgra8888 => PixelFormat::Bgra8888,
            ScalixPixelFormat::Rgb888 => PixelFormat::Rgb888,
            ScalixPixelFormat::Bgr888 => PixelFormat::Bgr888,
            ScalixPixelFormat::R8 => PixelFormat::R8,
            ScalixPixelFormat::Rg88 => PixelFormat::Rg88,
            ScalixPixelFormat::Nv12 => PixelFormat::Nv12,
            ScalixPixelFormat::Yuv420p => PixelFormat::Yuv420p,
            ScalixPixelFormat::Rgba16f => PixelFormat::Rgba16f,
            ScalixPixelFormat::Rgba32f => PixelFormat::Rgba32f,
        }
    }
}

#[repr(C)]
pub struct ScalixImageDesc {
    pub width: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub format: ScalixPixelFormat,
    pub host_ptr: *mut u8,
    pub data_len: usize,
    pub dma_buf_fd: i32,
}

pub type ScalixCompletionCallback =
    Option<unsafe extern "C" fn(status_code: i32, user_data: *mut c_void)>;

pub struct ScalixEngine {
    inner: Engine,
}

pub struct ScalixTask {
    inner: TaskHandle<OwnedImage>,
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_create(backend: ScalixBackendType) -> *mut ScalixEngine {
    scalix_engine_create_with_prefix(backend, std::ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_create_with_prefix(
    backend: ScalixBackendType,
    thread_prefix: *const std::os::raw::c_char,
) -> *mut ScalixEngine {
    let prefix = if thread_prefix.is_null() {
        None
    } else {
        std::ffi::CStr::from_ptr(thread_prefix).to_str().ok()
    };

    match Engine::with_prefix(backend.into(), prefix) {
        Ok(engine) => Box::into_raw(Box::new(ScalixEngine { inner: engine })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScalixProfileMetrics {
    pub host_unpack_ms: f64,
    pub gpu_upload_ms: f64,
    pub gpu_pure_blit_ms: f64,
    pub gpu_download_ms: f64,
    pub host_repack_ms: f64,
    pub driver_sync_ms: f64,
    pub total_wall_ms: f64,
}

impl From<scalix_core::ProfileMetrics> for ScalixProfileMetrics {
    fn from(m: scalix_core::ProfileMetrics) -> Self {
        Self {
            host_unpack_ms: m.host_unpack_ms,
            gpu_upload_ms: m.gpu_upload_ms,
            gpu_pure_blit_ms: m.gpu_pure_blit_ms,
            gpu_download_ms: m.gpu_download_ms,
            host_repack_ms: m.host_repack_ms,
            driver_sync_ms: m.driver_sync_ms,
            total_wall_ms: m.total_wall_ms,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_set_profiling(
    engine: *mut ScalixEngine,
    enabled: bool,
) -> i32 {
    if engine.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }
    (*engine).inner.set_profiling(enabled);
    SCALIX_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_get_last_profile(
    engine: *const ScalixEngine,
    out_metrics: *mut ScalixProfileMetrics,
) -> i32 {
    if engine.is_null() || out_metrics.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }
    match (*engine).inner.last_profile() {
        Some(m) => {
            *out_metrics = m.into();
            SCALIX_SUCCESS
        }
        None => SCALIX_ERR_FAILED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_destroy(engine: *mut ScalixEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_resize_sync(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *mut ScalixImageDesc,
    filter: ScalixFilterMode,
) -> i32 {
    if engine.is_null() || src.is_null() || dst.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let src = &*src;
    let dst = &mut *dst;

    if src.host_ptr.is_null() || dst.host_ptr.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let src_slice = slice::from_raw_parts(src.host_ptr, src.data_len);
    let dst_slice = slice::from_raw_parts_mut(dst.host_ptr, dst.data_len);

    let src_desc = match ImageDesc::new(
        src.width,
        src.height,
        src.stride_bytes,
        src.format.into(),
        src_slice,
    ) {
        Ok(d) => d,
        Err(e) => return map_error_to_code(e),
    };

    let mut dst_desc = match ImageDescMut::new(
        dst.width,
        dst.height,
        dst.stride_bytes,
        dst.format.into(),
        dst_slice,
    ) {
        Ok(d) => d,
        Err(e) => return map_error_to_code(e),
    };

    match (*engine).inner.resize_sync(&src_desc, &mut dst_desc, filter.into()) {
        Ok(()) => SCALIX_SUCCESS,
        Err(e) => map_error_to_code(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_resize_async(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    filter: ScalixFilterMode,
) -> *mut ScalixTask {
    if engine.is_null() || src.is_null() || dst.is_null() {
        return std::ptr::null_mut();
    }

    let src = &*src;
    let dst = &*dst;

    if src.host_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let src_slice = slice::from_raw_parts(src.host_ptr, src.data_len);
    let src_owned = match OwnedImage::from_vec(
        src.width,
        src.height,
        src.stride_bytes,
        src.format.into(),
        src_slice.to_vec(),
    ) {
        Ok(img) => img,
        Err(_) => return std::ptr::null_mut(),
    };

    let dst_owned = match OwnedImage::allocate(dst.width, dst.height, dst.format.into()) {
        Ok(img) => img,
        Err(_) => return std::ptr::null_mut(),
    };

    let task = (*engine).inner.resize_async(src_owned, dst_owned, filter.into());
    Box::into_raw(Box::new(ScalixTask { inner: task }))
}

#[no_mangle]
pub unsafe extern "C" fn scalix_task_is_ready(task: *const ScalixTask) -> bool {
    if task.is_null() {
        return false;
    }
    (*task).inner.is_ready()
}

#[no_mangle]
pub unsafe extern "C" fn scalix_task_wait(
    task: *mut ScalixTask,
    timeout_ms: u32,
    out_dst_ptr: *mut u8,
    out_dst_len: usize,
) -> i32 {
    if task.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let task = Box::from_raw(task);
    let timeout = if timeout_ms == 0 {
        None
    } else {
        Some(Duration::from_millis(timeout_ms as u64))
    };

    match task.inner.wait(timeout) {
        Ok(image) => {
            if !out_dst_ptr.is_null() {
                if out_dst_len < image.data.len() {
                    return SCALIX_ERR_BUFFER_TOO_SMALL;
                }
                std::ptr::copy_nonoverlapping(image.data.as_ptr(), out_dst_ptr, image.data.len());
            }
            SCALIX_SUCCESS
        }
        Err(e) => map_error_to_code(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_task_release(task: *mut ScalixTask) {
    if !task.is_null() {
        drop(Box::from_raw(task));
    }
}

// Wrapper struct for transmitting raw callback pointers across thread boundaries safely
struct CallbackCtx {
    callback: ScalixCompletionCallback,
    user_data: usize,
}
unsafe impl Send for CallbackCtx {}

#[no_mangle]
pub unsafe extern "C" fn scalix_resize_submit(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    filter: ScalixFilterMode,
    callback: ScalixCompletionCallback,
    user_data: *mut c_void,
) -> i32 {
    if engine.is_null() || src.is_null() || dst.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let src = &*src;
    let dst = &*dst;

    if src.host_ptr.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let src_slice = slice::from_raw_parts(src.host_ptr, src.data_len);
    let src_owned = match OwnedImage::from_vec(
        src.width,
        src.height,
        src.stride_bytes,
        src.format.into(),
        src_slice.to_vec(),
    ) {
        Ok(img) => img,
        Err(e) => return map_error_to_code(e),
    };

    let dst_owned = match OwnedImage::allocate(dst.width, dst.height, dst.format.into()) {
        Ok(img) => img,
        Err(e) => return map_error_to_code(e),
    };

    let ctx = CallbackCtx {
        callback,
        user_data: user_data as usize,
    };

    let res = (*engine).inner.resize_callback(
        src_owned,
        dst_owned,
        filter.into(),
        move |res| {
            let status = match res {
                Ok(_) => SCALIX_SUCCESS,
                Err(e) => map_error_to_code(e),
            };
            if let Some(cb) = ctx.callback {
                unsafe {
                    cb(status, ctx.user_data as *mut c_void);
                }
            }
        },
    );

    match res {
        Ok(()) => SCALIX_SUCCESS,
        Err(e) => map_error_to_code(e),
    }
}

pub struct ScalixDmaBuffer {
    inner: scalix_core::DmaBuffer,
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_allocate(
    width: u32,
    height: u32,
    format: ScalixPixelFormat,
) -> *mut ScalixDmaBuffer {
    match scalix_core::DmaBuffer::allocate(width, height, format.into()) {
        Ok(buf) => Box::into_raw(Box::new(ScalixDmaBuffer { inner: buf })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_free(buffer: *mut ScalixDmaBuffer) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_get_fd(buffer: *const ScalixDmaBuffer) -> i32 {
    if buffer.is_null() {
        return -1;
    }
    (*buffer).inner.fd()
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_get_host_ptr(buffer: *const ScalixDmaBuffer) -> *mut u8 {
    if buffer.is_null() {
        return std::ptr::null_mut();
    }
    (*buffer).inner.host_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_get_size(buffer: *const ScalixDmaBuffer) -> usize {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).inner.size()
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_get_stride(buffer: *const ScalixDmaBuffer) -> usize {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).inner.stride()
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_get_desc(
    buffer: *const ScalixDmaBuffer,
    out_desc: *mut ScalixImageDesc,
) -> i32 {
    if buffer.is_null() || out_desc.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }

    let b = &(*buffer).inner;
    let format = match b.format() {
        PixelFormat::Rgba8888 => ScalixPixelFormat::Rgba8888,
        PixelFormat::Bgra8888 => ScalixPixelFormat::Bgra8888,
        PixelFormat::Rgb888 => ScalixPixelFormat::Rgb888,
        PixelFormat::Bgr888 => ScalixPixelFormat::Bgr888,
        PixelFormat::R8 => ScalixPixelFormat::R8,
        PixelFormat::Rg88 => ScalixPixelFormat::Rg88,
        PixelFormat::Nv12 => ScalixPixelFormat::Nv12,
        PixelFormat::Yuv420p => ScalixPixelFormat::Yuv420p,
        PixelFormat::Rgba16f => ScalixPixelFormat::Rgba16f,
        PixelFormat::Rgba32f => ScalixPixelFormat::Rgba32f,
    };

    *out_desc = ScalixImageDesc {
        width: b.width(),
        height: b.height(),
        stride_bytes: b.stride(),
        format,
        host_ptr: b.host_ptr(),
        data_len: b.size(),
        dma_buf_fd: b.fd(),
    };

    SCALIX_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_sync_start(
    buffer: *const ScalixDmaBuffer,
    is_write: bool,
) -> i32 {
    if buffer.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }
    let flag = if is_write {
        scalix_core::DmaSyncFlags::Write
    } else {
        scalix_core::DmaSyncFlags::Read
    };
    match (*buffer).inner.sync_start(flag) {
        Ok(()) => SCALIX_SUCCESS,
        Err(e) => map_error_to_code(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_sync_end(
    buffer: *const ScalixDmaBuffer,
    is_write: bool,
) -> i32 {
    if buffer.is_null() {
        return SCALIX_ERR_NULL_PTR;
    }
    let flag = if is_write {
        scalix_core::DmaSyncFlags::Write
    } else {
        scalix_core::DmaSyncFlags::Read
    };
    match (*buffer).inner.sync_end(flag) {
        Ok(()) => SCALIX_SUCCESS,
        Err(e) => map_error_to_code(e),
    }
}

