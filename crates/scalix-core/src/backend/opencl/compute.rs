//! OpenCL Compute Resizer Pipeline (`clEnqueueNDRangeKernel`)
//!
//! Provides fused OpenCL C kernels operating directly on raw packed byte buffers
//! (RGB888, BGR888, RGBA8888, BGRA8888, R8, RG88) across all interpolation algorithms
//! (Nearest, Bilinear, Bicubic Catmull-Rom, Lanczos3, Area box average).

use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::backend::opencl::context::*;
use crate::backend::opencl::ring::{OpenClStagingRing, DEFAULT_OPENCL_RING_SLOTS};
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, PixelFormat, Result, ScalixError};

const SHADER_NEAREST: &str = include_str!("shaders/resize_nearest.cl");
const SHADER_BILINEAR: &str = include_str!("shaders/resize_bilinear.cl");
const SHADER_BICUBIC: &str = include_str!("shaders/resize_bicubic.cl");
const SHADER_LANCZOS3: &str = include_str!("shaders/resize_lanczos3.cl");
const SHADER_AREA: &str = include_str!("shaders/resize_area.cl");

struct CompiledKernel {
    program: ClProgram,
    kernel: ClKernel,
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

pub struct OpenClComputeResizer {
    ctx: Arc<OpenClContext>,
    ring: Mutex<OpenClStagingRing>,
    profiler: Arc<dyn Profiler>,
    kernels: Mutex<HashMap<FilterMode, CompiledKernel>>,
}

unsafe impl Send for OpenClComputeResizer {}
unsafe impl Sync for OpenClComputeResizer {}

impl OpenClComputeResizer {
    pub fn new(ctx: Arc<OpenClContext>, profiler: Arc<dyn Profiler>) -> Result<Self> {
        let ring = Mutex::new(OpenClStagingRing::new(
            Arc::clone(&ctx),
            DEFAULT_OPENCL_RING_SLOTS,
        )?);

        Ok(Self {
            ctx,
            ring,
            profiler,
            kernels: Mutex::new(HashMap::new()),
        })
    }

    fn get_or_build_kernel(&self, filter: FilterMode) -> Result<ClKernel> {
        let mut kernels_guard = self.kernels.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire OpenCL kernels lock".to_string())
        })?;

        if let Some(entry) = kernels_guard.get(&filter) {
            return Ok(entry.kernel);
        }

        let (src, c_kernel_name) = match filter {
            FilterMode::Nearest => (SHADER_NEAREST, c"resize_nearest"),
            FilterMode::Bilinear => (SHADER_BILINEAR, c"resize_bilinear"),
            FilterMode::Bicubic => (SHADER_BICUBIC, c"resize_bicubic"),
            FilterMode::Lanczos3 => (SHADER_LANCZOS3, c"resize_lanczos3"),
            FilterMode::Area => (SHADER_AREA, c"resize_area"),
            FilterMode::Passthrough => (SHADER_NEAREST, c"resize_nearest"),
        };

        let cl = &self.ctx.cl;
        let c_src = CString::new(src).map_err(|e| {
            ScalixError::ExecutionFailed(format!("Failed to create shader CString: {}", e))
        })?;
        let src_ptr = c_src.as_ptr();
        let src_len = src.len();

        let mut err: ClInt = 0;
        let program = unsafe {
            (cl.clCreateProgramWithSource)(self.ctx.context, 1, &src_ptr, &src_len, &mut err)
        };
        if err != CL_SUCCESS || program.is_null() {
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to create OpenCL program for {:?}: err={}",
                filter, err
            )));
        }

        let build_opts = c"-cl-fast-relaxed-math -cl-mad-enable";
        let build_err = unsafe {
            (cl.clBuildProgram)(
                program,
                1,
                &self.ctx.device,
                build_opts.as_ptr(),
                None,
                std::ptr::null_mut(),
            )
        };

        if build_err != CL_SUCCESS {
            let mut log_size: usize = 0;
            unsafe {
                (cl.clGetProgramBuildInfo)(
                    program,
                    self.ctx.device,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    std::ptr::null_mut(),
                    &mut log_size,
                )
            };
            let mut log_buf = vec![0u8; log_size.max(1)];
            if log_size > 0 {
                unsafe {
                    (cl.clGetProgramBuildInfo)(
                        program,
                        self.ctx.device,
                        CL_PROGRAM_BUILD_LOG,
                        log_size,
                        log_buf.as_mut_ptr() as *mut c_void,
                        std::ptr::null_mut(),
                    )
                };
            }
            let log_str = CStr::from_bytes_until_nul(&log_buf)
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            unsafe { (cl.clReleaseProgram)(program) };

            return Err(ScalixError::ExecutionFailed(format!(
                "OpenCL kernel build failed for {:?} (err={}):\nBuild Log:\n{}",
                filter, build_err, log_str
            )));
        }

        let kernel = unsafe { (cl.clCreateKernel)(program, c_kernel_name.as_ptr(), &mut err) };
        if err != CL_SUCCESS || kernel.is_null() {
            unsafe { (cl.clReleaseProgram)(program) };
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to create OpenCL kernel {:?}: err={}",
                c_kernel_name, err
            )));
        }

        kernels_guard.insert(filter, CompiledKernel { program, kernel });

        Ok(kernel)
    }

    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        if src.format != dst.format {
            return Err(ScalixError::ExecutionFailed(format!(
                "OpenCL direct compute resizer expects matching src and dst formats: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let channels = match src.format {
            PixelFormat::Rgba8888 | PixelFormat::Bgra8888 => 4u32,
            PixelFormat::Rgb888 | PixelFormat::Bgr888 => 3u32,
            PixelFormat::Rg88 => 2u32,
            PixelFormat::R8 => 1u32,
            PixelFormat::Rgba16f => 8u32,
            PixelFormat::Rgba32f => 16u32,
            PixelFormat::Nv12 | PixelFormat::Yuv420p => {
                return Err(ScalixError::UnsupportedFormat(src.format))
            }
        };

        let src_min_stride = (src.width as usize).saturating_mul(channels as usize);
        let dst_min_stride = (dst.width as usize).saturating_mul(channels as usize);
        if src.stride < src_min_stride || dst.stride < dst_min_stride {
            return Err(ScalixError::InvalidStride {
                stride: if src.stride < src_min_stride {
                    src.stride
                } else {
                    dst.stride
                },
                min_stride: if src.stride < src_min_stride {
                    src_min_stride
                } else {
                    dst_min_stride
                },
            });
        }

        let kernel = self.get_or_build_kernel(filter)?;
        let cl = &self.ctx.cl;
        let is_profiling = self.profiler.is_enabled();
        let wall_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire OpenClStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        let src_buf = slot.ensure_src_buffer(&self.ctx, src.data.len())?;
        let dst_buf = slot.ensure_dst_buffer(&self.ctx, dst.data.len())?;

        let _queue_guard =
            self.ctx.queue_lock.lock().map_err(|_| {
                ScalixError::ExecutionFailed("Failed to lock OpenCL queue".to_string())
            })?;

        // 1. Upload source data to GPU buffer
        let upload_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };
        let mut upload_event: ClEvent = std::ptr::null_mut();
        let err = unsafe {
            (cl.clEnqueueWriteBuffer)(
                self.ctx.queue,
                src_buf,
                CL_FALSE,
                0,
                src.data.len(),
                src.data.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                if is_profiling {
                    &mut upload_event
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        if err != CL_SUCCESS {
            return Err(ScalixError::ExecutionFailed(format!(
                "OpenCL clEnqueueWriteBuffer failed: {}",
                err
            )));
        }

        // 2. Set Kernel Arguments
        let src_w = src.width;
        let src_h = src.height;
        let dst_w = dst.width;
        let dst_h = dst.height;
        let src_stride = src.stride as u32;
        let dst_stride = dst.stride as u32;

        unsafe {
            let mut arg_idx = 0u32;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<ClMem>(),
                &src_buf as *const ClMem as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<ClMem>(),
                &dst_buf as *const ClMem as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &src_w as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &src_h as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &dst_w as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &dst_h as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &src_stride as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &dst_stride as *const u32 as *const c_void,
            );
            arg_idx += 1;
            (cl.clSetKernelArg)(
                kernel,
                arg_idx,
                std::mem::size_of::<u32>(),
                &channels as *const u32 as *const c_void,
            );
        }

        // 3. Dispatch Kernel
        let compute_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };
        let global_work_size: [usize; 2] = [dst_w as usize, dst_h as usize];
        let mut compute_event: ClEvent = std::ptr::null_mut();

        let err = unsafe {
            (cl.clEnqueueNDRangeKernel)(
                self.ctx.queue,
                kernel,
                2,
                std::ptr::null(),
                global_work_size.as_ptr(),
                std::ptr::null(), // Let driver choose optimal local workgroup size
                0,
                std::ptr::null(),
                if is_profiling {
                    &mut compute_event
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        if err != CL_SUCCESS {
            return Err(ScalixError::ExecutionFailed(format!(
                "OpenCL clEnqueueNDRangeKernel failed: {}",
                err
            )));
        }

        // 4. Download result from GPU buffer
        let download_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };
        let mut download_event: ClEvent = std::ptr::null_mut();
        let err = unsafe {
            (cl.clEnqueueReadBuffer)(
                self.ctx.queue,
                dst_buf,
                CL_BLOCKING,
                0,
                dst.data.len(),
                dst.data.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                if is_profiling {
                    &mut download_event
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        if err != CL_SUCCESS {
            return Err(ScalixError::ExecutionFailed(format!(
                "OpenCL clEnqueueReadBuffer failed: {}",
                err
            )));
        }

        if is_profiling {
            let wall_ms = wall_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let upload_ms = upload_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let compute_ms = compute_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let download_ms = download_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            // Cleanup profiling events
            unsafe {
                if !upload_event.is_null() {
                    (cl.clReleaseEvent)(upload_event);
                }
                if !compute_event.is_null() {
                    (cl.clReleaseEvent)(compute_event);
                }
                if !download_event.is_null() {
                    (cl.clReleaseEvent)(download_event);
                }
            }

            self.profiler.record(ProfileMetrics {
                host_unpack_ms: 0.0,
                gpu_upload_ms: upload_ms,
                gpu_pure_blit_ms: compute_ms,
                gpu_download_ms: download_ms,
                host_repack_ms: 0.0,
                driver_sync_ms: 0.0,
                total_wall_ms: wall_ms,
            });
        }

        Ok(())
    }
}

impl Drop for OpenClComputeResizer {
    fn drop(&mut self) {
        let cl = &self.ctx.cl;
        if let Ok(mut guard) = self.kernels.lock() {
            for (_, entry) in guard.drain() {
                unsafe {
                    (cl.clReleaseKernel)(entry.kernel);
                    (cl.clReleaseProgram)(entry.program);
                }
            }
        }
    }
}
