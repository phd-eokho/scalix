#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;
use std::slice;
use std::time::Duration;

use scalix_core::{
    BackendType, Engine, FilterMode, ImageDesc, ImageDescMut, OwnedImage, PixelFormat, Result,
    ScalixError, TaskHandle,
};

macro_rules! ffi_catch {
    ($fallback:expr, $body:block) => {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)).unwrap_or($fallback)
    };
    ($body:block) => {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
    };
}

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
pub const SCALIX_ERR_UNALIGNED_POINTER: i32 = -12;
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
        ScalixError::UnalignedPointer { .. } => SCALIX_ERR_UNALIGNED_POINTER,
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
pub enum ScalixStrategy {
    Auto = 0,
    Blit = 1,
    Raster = 2,
    LodPyramid = 3,
    Compute = 4,
}

impl From<ScalixStrategy> for scalix_core::VulkanStrategy {
    fn from(s: ScalixStrategy) -> Self {
        match s {
            ScalixStrategy::Auto => scalix_core::VulkanStrategy::Auto,
            ScalixStrategy::Blit => scalix_core::VulkanStrategy::Blit,
            ScalixStrategy::Raster => scalix_core::VulkanStrategy::Raster,
            ScalixStrategy::LodPyramid => scalix_core::VulkanStrategy::LodPyramid,
            ScalixStrategy::Compute => scalix_core::VulkanStrategy::Compute,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalixBackendOptions {
    pub backend_type: ScalixBackendType,
    pub struct_size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalixVulkanOptions {
    pub header: ScalixBackendOptions,
    pub strategy: ScalixStrategy,
    pub max_mip_levels: u32,
}

impl Default for ScalixVulkanOptions {
    fn default() -> Self {
        Self {
            header: ScalixBackendOptions {
                backend_type: ScalixBackendType::Vulkan,
                struct_size: std::mem::size_of::<Self>() as u32,
            },
            strategy: ScalixStrategy::Auto,
            max_mip_levels: 0,
        }
    }
}

impl From<ScalixVulkanOptions> for scalix_core::VulkanOptions {
    fn from(v: ScalixVulkanOptions) -> Self {
        Self {
            strategy: v.strategy.into(),
            max_mip_levels: v.max_mip_levels,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalixAllocatorType {
    Auto = 0,
    DmaHeap = 1,
    DrmDumb = 2,
    AndroidAhb = 3,
    HostAligned = 4,
}

impl From<ScalixAllocatorType> for scalix_core::DmaAllocatorType {
    fn from(a: ScalixAllocatorType) -> Self {
        match a {
            ScalixAllocatorType::Auto => scalix_core::DmaAllocatorType::Auto,
            ScalixAllocatorType::DmaHeap => scalix_core::DmaAllocatorType::DmaHeap,
            ScalixAllocatorType::DrmDumb => scalix_core::DmaAllocatorType::DrmDumb,
            ScalixAllocatorType::AndroidAhb => scalix_core::DmaAllocatorType::AndroidAhb,
            ScalixAllocatorType::HostAligned => scalix_core::DmaAllocatorType::HostAligned,
        }
    }
}

impl From<scalix_core::DmaAllocatorType> for ScalixAllocatorType {
    fn from(a: scalix_core::DmaAllocatorType) -> Self {
        match a {
            scalix_core::DmaAllocatorType::Auto => ScalixAllocatorType::Auto,
            scalix_core::DmaAllocatorType::DmaHeap => ScalixAllocatorType::DmaHeap,
            scalix_core::DmaAllocatorType::DrmDumb => ScalixAllocatorType::DrmDumb,
            scalix_core::DmaAllocatorType::AndroidAhb => ScalixAllocatorType::AndroidAhb,
            scalix_core::DmaAllocatorType::HostAligned => ScalixAllocatorType::HostAligned,
        }
    }
}

impl From<ScalixStrategy> for scalix_core::GlStrategy {
    fn from(s: ScalixStrategy) -> Self {
        match s {
            ScalixStrategy::Auto => scalix_core::GlStrategy::Auto,
            ScalixStrategy::Blit => scalix_core::GlStrategy::Blit,
            ScalixStrategy::Raster => scalix_core::GlStrategy::Raster,
            ScalixStrategy::LodPyramid => scalix_core::GlStrategy::LodPyramid,
            ScalixStrategy::Compute => scalix_core::GlStrategy::Compute,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalixGlOptions {
    pub header: ScalixBackendOptions,
    pub strategy: ScalixStrategy,
    pub max_mip_levels: u32,
}

impl Default for ScalixGlOptions {
    fn default() -> Self {
        Self {
            header: ScalixBackendOptions {
                backend_type: ScalixBackendType::OpenGL,
                struct_size: std::mem::size_of::<Self>() as u32,
            },
            strategy: ScalixStrategy::Auto,
            max_mip_levels: 0,
        }
    }
}

impl From<ScalixGlOptions> for scalix_core::GlOptions {
    fn from(g: ScalixGlOptions) -> Self {
        Self {
            strategy: g.strategy.into(),
            max_mip_levels: g.max_mip_levels,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalixResizeOptions {
    pub filter: ScalixFilterMode,
    pub backend_options: *const ScalixBackendOptions,
}

impl ScalixResizeOptions {
    /// Safely convert C ABI `ScalixResizeOptions` to core `ResizeOptions`.
    ///
    /// # Safety
    /// If `self.backend_options` is non-null, it must point to a valid struct starting with
    /// `ScalixBackendOptions` header whose `struct_size` matches or exceeds the backend-specific struct size.
    pub unsafe fn to_core(&self) -> scalix_core::ResizeOptions {
        let mut core_options = scalix_core::ResizeOptions::new(self.filter.into());
        if !self.backend_options.is_null() {
            let header = &*self.backend_options;
            match header.backend_type {
                ScalixBackendType::Vulkan
                    if header.struct_size as usize
                        >= std::mem::size_of::<ScalixVulkanOptions>() =>
                {
                    let vk = &*(self.backend_options as *const ScalixVulkanOptions);
                    core_options = core_options.with_vulkan_options(scalix_core::VulkanOptions {
                        strategy: vk.strategy.into(),
                        max_mip_levels: vk.max_mip_levels,
                    });
                }
                ScalixBackendType::OpenGL
                    if header.struct_size as usize >= std::mem::size_of::<ScalixGlOptions>() =>
                {
                    let gl = &*(self.backend_options as *const ScalixGlOptions);
                    core_options = core_options.with_gl_options(scalix_core::GlOptions {
                        strategy: gl.strategy.into(),
                        max_mip_levels: gl.max_mip_levels,
                    });
                }
                _ => {}
            }
        }
        core_options
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
    inner_owned: Option<TaskHandle<OwnedImage>>,
    inner_raw: Option<TaskHandle<()>>,
    result: Option<OwnedImage>,
}

unsafe fn extract_image_desc<'a>(desc: *const ScalixImageDesc) -> Result<ImageDesc<'a>> {
    if desc.is_null() {
        return Err(ScalixError::NullPointer);
    }
    let d = &*desc;
    if d.host_ptr.is_null() {
        return Err(ScalixError::NullPointer);
    }
    let slice = slice::from_raw_parts(d.host_ptr, d.data_len);
    let mut image_desc = ImageDesc::new(d.width, d.height, d.stride_bytes, d.format.into(), slice)?;
    if d.dma_buf_fd >= 0 {
        image_desc = image_desc.with_dma_buf(d.dma_buf_fd);
    }
    Ok(image_desc)
}

unsafe fn extract_image_desc_mut<'a>(desc: *mut ScalixImageDesc) -> Result<ImageDescMut<'a>> {
    if desc.is_null() {
        return Err(ScalixError::NullPointer);
    }
    let d = &mut *desc;
    if d.host_ptr.is_null() {
        return Err(ScalixError::NullPointer);
    }
    let slice = slice::from_raw_parts_mut(d.host_ptr, d.data_len);
    let mut image_desc =
        ImageDescMut::new(d.width, d.height, d.stride_bytes, d.format.into(), slice)?;
    if d.dma_buf_fd >= 0 {
        image_desc = image_desc.with_dma_buf(d.dma_buf_fd);
    }
    Ok(image_desc)
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_engine_create(backend: ScalixBackendType) -> *mut ScalixEngine {
    ffi_catch!(std::ptr::null_mut(), {
        scalix_engine_create_with_prefix(backend, std::ptr::null())
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_engine_create_with_prefix(
    backend: ScalixBackendType,
    thread_prefix: *const std::os::raw::c_char,
) -> *mut ScalixEngine {
    ffi_catch!(std::ptr::null_mut(), {
        let prefix = if thread_prefix.is_null() {
            None
        } else {
            std::ffi::CStr::from_ptr(thread_prefix).to_str().ok()
        };

        match Engine::with_prefix(backend.into(), prefix) {
            Ok(engine) => Box::into_raw(Box::new(ScalixEngine { inner: engine })),
            Err(_) => std::ptr::null_mut(),
        }
    })
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
#[must_use]
pub unsafe extern "C" fn scalix_engine_get_backend_name(
    engine: *const ScalixEngine,
) -> *const std::ffi::c_char {
    ffi_catch!(std::ptr::null(), {
        if engine.is_null() {
            return std::ptr::null();
        }
        match (*engine).inner.backend_type() {
            BackendType::Vulkan => c"Vulkan".as_ptr() as *const std::ffi::c_char,
            BackendType::OpenGL => c"OpenGL".as_ptr() as *const std::ffi::c_char,
            BackendType::Npu => c"NPU".as_ptr() as *const std::ffi::c_char,
            BackendType::Hw2d => c"Hardware 2D".as_ptr() as *const std::ffi::c_char,
            BackendType::Cpu => c"CPU (Fallback)".as_ptr() as *const std::ffi::c_char,
            BackendType::Passthrough => c"Passthrough".as_ptr() as *const std::ffi::c_char,
            BackendType::Auto => c"Auto".as_ptr() as *const std::ffi::c_char,
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_engine_get_backend_type(
    engine: *const ScalixEngine,
) -> ScalixBackendType {
    ffi_catch!(ScalixBackendType::Auto, {
        if engine.is_null() {
            return ScalixBackendType::Auto;
        }
        match (*engine).inner.backend_type() {
            BackendType::Auto => ScalixBackendType::Auto,
            BackendType::Vulkan => ScalixBackendType::Vulkan,
            BackendType::OpenGL => ScalixBackendType::OpenGL,
            BackendType::Npu => ScalixBackendType::Npu,
            BackendType::Hw2d => ScalixBackendType::Hw2d,
            BackendType::Cpu => ScalixBackendType::Cpu,
            BackendType::Passthrough => ScalixBackendType::Passthrough,
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_engine_set_profiling(
    engine: *mut ScalixEngine,
    enabled: bool,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        if engine.is_null() {
            return SCALIX_ERR_NULL_PTR;
        }
        (*engine).inner.set_profiling(enabled);
        SCALIX_SUCCESS
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_engine_get_last_profile(
    engine: *const ScalixEngine,
    out_metrics: *mut ScalixProfileMetrics,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
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
    })
}

#[no_mangle]
pub unsafe extern "C" fn scalix_engine_destroy(engine: *mut ScalixEngine) {
    ffi_catch!({
        if !engine.is_null() {
            drop(Box::from_raw(engine));
        }
    });
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_sync_with_options(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *mut ScalixImageDesc,
    options: *const ScalixResizeOptions,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        if engine.is_null() || options.is_null() {
            return SCALIX_ERR_NULL_PTR;
        }

        let src_desc = match extract_image_desc(src) {
            Ok(d) => d,
            Err(e) => return map_error_to_code(e),
        };

        let mut dst_desc = match extract_image_desc_mut(dst) {
            Ok(d) => d,
            Err(e) => return map_error_to_code(e),
        };

        let core_options = (*options).to_core();
        match (*engine)
            .inner
            .resize_sync(&src_desc, &mut dst_desc, core_options)
        {
            Ok(()) => SCALIX_SUCCESS,
            Err(e) => map_error_to_code(e),
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_sync(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *mut ScalixImageDesc,
    filter: ScalixFilterMode,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        let options = ScalixResizeOptions {
            filter,
            backend_options: std::ptr::null(),
        };
        scalix_resize_sync_with_options(engine, src, dst, &options)
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_async_with_options(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    options: *const ScalixResizeOptions,
) -> *mut ScalixTask {
    ffi_catch!(std::ptr::null_mut(), {
        if engine.is_null() || src.is_null() || dst.is_null() || options.is_null() {
            return std::ptr::null_mut();
        }

        let src_ref = &*src;
        let dst_ref = &*dst;
        let options = &*options;

        if src_ref.host_ptr.is_null() {
            return std::ptr::null_mut();
        }

        let core_options = (*options).to_core();

        if !dst_ref.host_ptr.is_null() {
            // Direct zero-copy path: both src and dst host pointers are provided upfront
            let src_desc = match extract_image_desc(src) {
                Ok(d) => std::mem::transmute::<ImageDesc<'_>, ImageDesc<'static>>(d),
                Err(_) => return std::ptr::null_mut(),
            };

            let dst_desc = match extract_image_desc_mut(dst as *mut ScalixImageDesc) {
                Ok(d) => std::mem::transmute::<ImageDescMut<'_>, ImageDescMut<'static>>(d),
                Err(_) => return std::ptr::null_mut(),
            };

            let task = (*engine)
                .inner
                .resize_async_raw(src_desc, dst_desc, core_options);

            Box::into_raw(Box::new(ScalixTask {
                inner_owned: None,
                inner_raw: Some(task),
                result: None,
            }))
        } else {
            // Deferred destination path: dst_ref.host_ptr is null, allocate intermediate buffer
            let src_slice = slice::from_raw_parts(src_ref.host_ptr, src_ref.data_len);
            let src_owned = match OwnedImage::from_slice(
                src_ref.width,
                src_ref.height,
                src_ref.stride_bytes,
                src_ref.format.into(),
                src_slice,
            ) {
                Ok(img) => img,
                Err(_) => return std::ptr::null_mut(),
            };

            let dst_owned =
                match OwnedImage::allocate(dst_ref.width, dst_ref.height, dst_ref.format.into()) {
                    Ok(img) => img,
                    Err(_) => return std::ptr::null_mut(),
                };

            let task = (*engine)
                .inner
                .resize_async(src_owned, dst_owned, core_options);

            Box::into_raw(Box::new(ScalixTask {
                inner_owned: Some(task),
                inner_raw: None,
                result: None,
            }))
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_async(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    filter: ScalixFilterMode,
) -> *mut ScalixTask {
    ffi_catch!(std::ptr::null_mut(), {
        let options = ScalixResizeOptions {
            filter,
            backend_options: std::ptr::null(),
        };
        scalix_resize_async_with_options(engine, src, dst, &options)
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_task_is_ready(task: *const ScalixTask) -> bool {
    ffi_catch!(false, {
        if task.is_null() {
            return false;
        }
        let task_ref = &*task;
        if task_ref.result.is_some() {
            return true;
        }
        if let Some(ref inner) = task_ref.inner_raw {
            return inner.is_ready();
        }
        if let Some(ref inner) = task_ref.inner_owned {
            return inner.is_ready();
        }
        false
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_task_wait(
    task: *mut ScalixTask,
    timeout_ms: u32,
    out_dst_ptr: *mut u8,
    out_dst_len: usize,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        if task.is_null() {
            return SCALIX_ERR_NULL_PTR;
        }

        let task_ref = &mut *task;
        let timeout = if timeout_ms == 0 || timeout_ms == u32::MAX {
            None
        } else {
            Some(Duration::from_millis(timeout_ms as u64))
        };

        if let Some(inner) = task_ref.inner_raw.take() {
            match inner.wait(timeout) {
                Ok(()) => return SCALIX_SUCCESS,
                Err(ScalixError::Timeout) => {
                    task_ref.inner_raw = Some(inner);
                    return SCALIX_ERR_TIMEOUT;
                }
                Err(e) => return map_error_to_code(e),
            }
        }

        let image = if let Some(ref img) = task_ref.result {
            img
        } else if let Some(inner) = task_ref.inner_owned.take() {
            match inner.wait(timeout) {
                Ok(img) => {
                    task_ref.result = Some(img);
                    task_ref.result.as_ref().unwrap()
                }
                Err(ScalixError::Timeout) => {
                    task_ref.inner_owned = Some(inner);
                    return SCALIX_ERR_TIMEOUT;
                }
                Err(e) => return map_error_to_code(e),
            }
        } else {
            return SCALIX_ERR_FAILED;
        };

        if !out_dst_ptr.is_null() {
            if out_dst_len < image.data.as_slice().len() {
                return SCALIX_ERR_BUFFER_TOO_SMALL;
            }
            std::ptr::copy_nonoverlapping(
                image.data.as_ptr(),
                out_dst_ptr,
                image.data.as_slice().len(),
            );
        }
        SCALIX_SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn scalix_task_release(task: *mut ScalixTask) {
    ffi_catch!({
        if !task.is_null() {
            drop(Box::from_raw(task));
        }
    });
}

// Wrapper struct for transmitting raw callback pointers across thread boundaries safely
struct CallbackCtx {
    callback: ScalixCompletionCallback,
    user_data: usize,
}
unsafe impl Send for CallbackCtx {}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_submit_with_options(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    options: *const ScalixResizeOptions,
    callback: ScalixCompletionCallback,
    user_data: *mut c_void,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        if engine.is_null() || src.is_null() || dst.is_null() || options.is_null() {
            return SCALIX_ERR_NULL_PTR;
        }

        let src = &*src;
        let dst = &*dst;
        let options = &*options;

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

        let core_options = (*options).to_core();
        let res =
            (*engine)
                .inner
                .resize_callback(
                    src_owned,
                    dst_owned,
                    core_options,
                    move |task_res| match task_res {
                        Ok(_img) => {
                            if let Some(cb) = ctx.callback {
                                unsafe { cb(SCALIX_SUCCESS, ctx.user_data as *mut c_void) };
                            }
                        }
                        Err(e) => {
                            if let Some(cb) = ctx.callback {
                                unsafe { cb(map_error_to_code(e), ctx.user_data as *mut c_void) };
                            }
                        }
                    },
                );

        match res {
            Ok(()) => SCALIX_SUCCESS,
            Err(e) => map_error_to_code(e),
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_resize_submit(
    engine: *mut ScalixEngine,
    src: *const ScalixImageDesc,
    dst: *const ScalixImageDesc,
    filter: ScalixFilterMode,
    callback: ScalixCompletionCallback,
    user_data: *mut c_void,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
        let options = ScalixResizeOptions {
            filter,
            backend_options: std::ptr::null(),
        };
        scalix_resize_submit_with_options(engine, src, dst, &options, callback, user_data)
    })
}

pub struct ScalixDmaBuffer {
    inner: scalix_core::DmaBuffer,
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_allocate(
    width: u32,
    height: u32,
    format: ScalixPixelFormat,
) -> *mut ScalixDmaBuffer {
    ffi_catch!(std::ptr::null_mut(), {
        match scalix_core::DmaBuffer::allocate(width, height, format.into()) {
            Ok(buf) => Box::into_raw(Box::new(ScalixDmaBuffer { inner: buf })),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_allocate_with_type(
    width: u32,
    height: u32,
    format: ScalixPixelFormat,
    allocator_type: ScalixAllocatorType,
) -> *mut ScalixDmaBuffer {
    ffi_catch!(std::ptr::null_mut(), {
        match scalix_core::DmaBuffer::allocate_with_type(
            width,
            height,
            format.into(),
            allocator_type.into(),
        ) {
            Ok(buf) => Box::into_raw(Box::new(ScalixDmaBuffer { inner: buf })),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_allocator_type(
    buffer: *const ScalixDmaBuffer,
) -> ScalixAllocatorType {
    ffi_catch!(ScalixAllocatorType::Auto, {
        if buffer.is_null() {
            return ScalixAllocatorType::Auto;
        }
        (*buffer).inner.allocator_type().into()
    })
}

#[no_mangle]
pub unsafe extern "C" fn scalix_dma_buffer_free(buffer: *mut ScalixDmaBuffer) {
    ffi_catch!({
        if !buffer.is_null() {
            drop(Box::from_raw(buffer));
        }
    });
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_fd(buffer: *const ScalixDmaBuffer) -> i32 {
    ffi_catch!(-1, {
        if buffer.is_null() {
            return -1;
        }
        (*buffer).inner.fd()
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_host_ptr(buffer: *const ScalixDmaBuffer) -> *mut u8 {
    ffi_catch!(std::ptr::null_mut(), {
        if buffer.is_null() {
            return std::ptr::null_mut();
        }
        (*buffer).inner.host_ptr()
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_size(buffer: *const ScalixDmaBuffer) -> usize {
    ffi_catch!(0, {
        if buffer.is_null() {
            return 0;
        }
        (*buffer).inner.size()
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_stride(buffer: *const ScalixDmaBuffer) -> usize {
    ffi_catch!(0, {
        if buffer.is_null() {
            return 0;
        }
        (*buffer).inner.stride()
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_get_desc(
    buffer: *const ScalixDmaBuffer,
    out_desc: *mut ScalixImageDesc,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
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
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_sync_start(
    buffer: *const ScalixDmaBuffer,
    is_write: bool,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
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
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn scalix_dma_buffer_sync_end(
    buffer: *const ScalixDmaBuffer,
    is_write: bool,
) -> i32 {
    ffi_catch!(SCALIX_ERR_FAILED, {
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
    })
}
