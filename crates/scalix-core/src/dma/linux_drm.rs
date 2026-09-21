//! Linux DRM Render Node Dumb Buffer Allocator (`/dev/dri/renderD128`, GEM PRIME)

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use crate::types::{PixelFormat, Result, ScalixError};

#[repr(C)]
struct DrmModeCreateDumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}

#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
struct DrmModeDestroyDumb {
    handle: u32,
}

const DRM_IOCTL_MODE_CREATE_DUMB: libc::c_ulong = 0xc02064b2;
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::c_ulong = 0xc00c642d;
const DRM_IOCTL_MODE_DESTROY_DUMB: libc::c_ulong = 0xc00464b4;

const DRM_CANDIDATE_PATHS: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
    "/dev/dri/card0",
];

pub struct LinuxDrmAllocator;

impl LinuxDrmAllocator {
    /// Attempts to probe and allocate a DMA-BUF file descriptor from available DRM render nodes.
    pub fn allocate(
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<(OwnedFd, usize, usize)> {
        let bpp = match format {
            PixelFormat::Rgba8888 | PixelFormat::Bgra8888 => 32,
            PixelFormat::Rgb888 | PixelFormat::Bgr888 => 24,
            PixelFormat::R8 => 8,
            PixelFormat::Rg88 => 16,
            PixelFormat::Rgba16f => 64,
            PixelFormat::Rgba32f => 128,
            PixelFormat::Nv12 | PixelFormat::Yuv420p => 8, // 8bpp base
        };

        let mut last_err = String::from("No DRM render nodes accessible");

        for &path in DRM_CANDIDATE_PATHS {
            let file = match OpenOptions::new().read(true).write(true).open(path) {
                Ok(f) => f,
                Err(e) => {
                    last_err = format!("{}: {}", path, e);
                    continue;
                }
            };

            let drm_fd = file.as_raw_fd();

            // 1. Create dumb buffer
            let mut create_dumb = DrmModeCreateDumb {
                width,
                height: if format == PixelFormat::Nv12 || format == PixelFormat::Yuv420p {
                    height * 3 / 2
                } else {
                    height
                },
                bpp,
                flags: 0,
                handle: 0,
                pitch: 0,
                size: 0,
            };

            let ret = unsafe {
                libc::ioctl(
                    drm_fd,
                    DRM_IOCTL_MODE_CREATE_DUMB,
                    &mut create_dumb as *mut DrmModeCreateDumb,
                )
            };

            if ret != 0 {
                let errno = std::io::Error::last_os_error();
                last_err = format!("DRM_IOCTL_MODE_CREATE_DUMB on {} failed: {}", path, errno);
                continue;
            }

            let handle = create_dumb.handle;
            let pitch = create_dumb.pitch as usize;
            let size = create_dumb.size as usize;

            // 2. Export GEM handle to DMA-BUF PRIME fd
            let mut prime_handle = DrmPrimeHandle {
                handle,
                flags: libc::O_CLOEXEC as u32 | libc::O_RDWR as u32,
                fd: -1,
            };

            let ret_prime = unsafe {
                libc::ioctl(
                    drm_fd,
                    DRM_IOCTL_PRIME_HANDLE_TO_FD,
                    &mut prime_handle as *mut DrmPrimeHandle,
                )
            };

            // Release GEM handle reference on DRM device (the DMA-BUF fd holds its own ref)
            let mut destroy_dumb = DrmModeDestroyDumb { handle };
            unsafe {
                libc::ioctl(
                    drm_fd,
                    DRM_IOCTL_MODE_DESTROY_DUMB,
                    &mut destroy_dumb as *mut DrmModeDestroyDumb,
                );
            }

            if ret_prime == 0 && prime_handle.fd >= 0 {
                let dma_fd = unsafe { OwnedFd::from_raw_fd(prime_handle.fd as RawFd) };
                return Ok((dma_fd, size, pitch));
            } else {
                let errno = std::io::Error::last_os_error();
                last_err = format!("DRM_IOCTL_PRIME_HANDLE_TO_FD on {} failed: {}", path, errno);
            }
        }

        Err(ScalixError::DmaUnavailable(last_err))
    }
}
