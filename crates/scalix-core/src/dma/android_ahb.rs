//! Android AHardwareBuffer Allocator (Strictly `aarch64-linux-android` and API 26+)

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
use {
    std::os::fd::{FromRawFd, OwnedFd, RawFd},
    std::ptr::NonNull,
    crate::types::{PixelFormat, Result, ScalixError},
};

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
#[repr(C)]
struct AHardwareBuffer_Desc {
    width: u32,
    height: u32,
    layers: u32,
    format: u32,
    usage: u64,
    stride: u32,
    rfu0: u32,
    rfu1: u64,
}

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM: u32 = 1;
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM: u32 = 3;
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420: u32 = 0x23; // NV12 / YUV420

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 0x02;
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 0x20;
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE: u64 = 0x100;
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT: u64 = 0x200;

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
#[link(name = "android")]
extern "C" {
    fn AHardwareBuffer_allocate(
        desc: *const AHardwareBuffer_Desc,
        outBuffer: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn AHardwareBuffer_release(buffer: *mut std::ffi::c_void);
    fn AHardwareBuffer_lock(
        buffer: *mut std::ffi::c_void,
        usage: u64,
        fence: i32,
        rect: *const std::ffi::c_void,
        outVirtualAddress: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn AHardwareBuffer_unlock(buffer: *mut std::ffi::c_void, fence: *mut i32) -> i32;
}

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
pub struct AndroidAhbAllocator;

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
impl AndroidAhbAllocator {
    pub fn allocate(
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<(*mut std::ffi::c_void, *mut u8, usize, usize)> {
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

        let mut desc = AHardwareBuffer_Desc {
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
                "AHardwareBuffer_allocate failed with status: {}",
                status
            )));
        }

        let mut vaddr: *mut std::ffi::c_void = std::ptr::null_mut();
        let lock_status = unsafe {
            AHardwareBuffer_lock(
                ahb_ptr,
                usage,
                -1,
                std::ptr::null(),
                &mut vaddr,
            )
        };

        if lock_status != 0 || vaddr.is_null() {
            unsafe { AHardwareBuffer_release(ahb_ptr) };
            return Err(ScalixError::DmaMapFailed(format!(
                "AHardwareBuffer_lock failed with status: {}",
                lock_status
            )));
        }

        let stride = format.min_stride(width)?;
        let size = format.min_buffer_size(width, height, stride)?;

        Ok((ahb_ptr, vaddr as *mut u8, size, stride))
    }
}
