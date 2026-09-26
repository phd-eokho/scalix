//! Android AHardwareBuffer Allocator (`aarch64` and `armv7` on Android API 26+)

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
use {
    crate::types::{ImageDimensions, PixelFormat, Result, ScalixError},
    std::os::fd::{FromRawFd, OwnedFd, RawFd},
    std::ptr::NonNull,
};

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
#[repr(C)]
pub(crate) struct AHardwareBuffer_Desc {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) format: u32,
    pub(crate) usage: u64,
    pub(crate) stride: u32,
    pub(crate) rfu0: u32,
    pub(crate) rfu1: u64,
}

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM: u32 = 1;
#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM: u32 = 3;
#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420: u32 = 0x23; // NV12 / YUV420

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 0x02;
#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 0x20;
#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE: u64 = 0x100;
#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
const AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT: u64 = 0x200;

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
#[link(name = "android")]
extern "C" {
    pub(crate) fn AHardwareBuffer_allocate(
        desc: *const AHardwareBuffer_Desc,
        outBuffer: *mut *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn AHardwareBuffer_describe(
        buffer: *const std::ffi::c_void,
        desc: *mut AHardwareBuffer_Desc,
    );
    pub(crate) fn AHardwareBuffer_release(buffer: *mut std::ffi::c_void);
    pub(crate) fn AHardwareBuffer_lock(
        buffer: *mut std::ffi::c_void,
        usage: u64,
        fence: i32,
        rect: *const std::ffi::c_void,
        outVirtualAddress: *mut *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn AHardwareBuffer_unlock(buffer: *mut std::ffi::c_void, fence: *mut i32) -> i32;
}

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub struct AndroidAhbAllocator;

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
impl AndroidAhbAllocator {
    #[inline]
    #[must_use]
    pub fn is_available() -> bool {
        true
    }

    #[inline]
    pub fn allocate(
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<(*mut std::ffi::c_void, *mut u8, usize, usize)> {
        Self::allocate_dimensions(ImageDimensions::new(width, height), format)
    }

    pub fn allocate_dimensions(
        dimensions: ImageDimensions,
        format: PixelFormat,
    ) -> Result<(*mut std::ffi::c_void, *mut u8, usize, usize)> {
        let (width, height) = (dimensions.width, dimensions.height);
        let ahb_format = match format {
            PixelFormat::Rgba8888 => AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
            PixelFormat::Rgb888 => AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
            PixelFormat::Nv12 | PixelFormat::Yuv420p => AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420,
            other => return Err(ScalixError::UnsupportedFormat(other)),
        };

        let usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
            | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN
            | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
            | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;

        let desc = AHardwareBuffer_Desc {
            width,
            height,
            layers: 1,
            format: ahb_format,
            usage,
            stride: 0,
            rfu0: 0,
            rfu1: 0,
        };

        let mut ahb_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { AHardwareBuffer_allocate(&desc, &mut ahb_ptr) };
        if status != 0 || ahb_ptr.is_null() {
            return Err(ScalixError::DmaAllocationFailed(format!(
                "AHardwareBuffer_allocate failed with status: {status}"
            )));
        }

        let mut out_desc = AHardwareBuffer_Desc {
            width: 0,
            height: 0,
            layers: 0,
            format: 0,
            usage: 0,
            stride: 0,
            rfu0: 0,
            rfu1: 0,
        };
        unsafe { AHardwareBuffer_describe(ahb_ptr, &mut out_desc) };

        let mut vaddr: *mut std::ffi::c_void = std::ptr::null_mut();
        let lock_status =
            unsafe { AHardwareBuffer_lock(ahb_ptr, usage, -1, std::ptr::null(), &mut vaddr) };

        if lock_status != 0 || vaddr.is_null() {
            unsafe { AHardwareBuffer_release(ahb_ptr) };
            return Err(ScalixError::DmaMapFailed(format!(
                "AHardwareBuffer_lock failed with status: {lock_status}"
            )));
        }

        let min_stride = format.min_stride(width)?;
        let bpp = format.bytes_per_pixel().unwrap_or(1);
        let actual_stride = (out_desc.stride as usize)
            .saturating_mul(bpp)
            .max(min_stride);
        let size = format.min_buffer_size_dims(dimensions, actual_stride)?;

        Ok((ahb_ptr, vaddr as *mut u8, size, actual_stride))
    }
}
