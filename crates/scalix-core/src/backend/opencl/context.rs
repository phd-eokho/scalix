//! Dynamic OpenCL ICD Loader and Context Management
//!
//! Provides headless, runtime-loaded OpenCL 1.2+ context discovery and device initialization
//! supporting desktop Linux, WSL2 (Rusticl / D3D12), and Android vendor drivers (Adreno, Mali, PowerVR).

use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex};

use crate::types::{Result, ScalixError};

pub type ClPlatformId = *mut c_void;
pub type ClDeviceId = *mut c_void;
pub type ClContext = *mut c_void;
pub type ClCommandQueue = *mut c_void;
pub type ClMem = *mut c_void;
pub type ClProgram = *mut c_void;
pub type ClKernel = *mut c_void;
pub type ClEvent = *mut c_void;
pub type ClInt = i32;
pub type ClUint = u32;
pub type ClUlong = u64;
pub type ClBool = u32;
pub type ClDeviceType = u64;
pub type ClPlatformInfo = u32;
pub type ClDeviceInfo = u32;
pub type ClMemFlags = u64;
pub type ClCommandQueueProperties = u64;
pub type ClProgramBuildInfo = u32;
pub type ClProfilingInfo = u32;

pub const CL_SUCCESS: ClInt = 0;
pub const CL_DEVICE_TYPE_DEFAULT: ClDeviceType = 1 << 0;
pub const CL_DEVICE_TYPE_CPU: ClDeviceType = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR: ClDeviceType = 1 << 3;
pub const CL_DEVICE_TYPE_ALL: ClDeviceType = 0xFFFFFFFF;

pub const CL_PLATFORM_NAME: ClPlatformInfo = 0x0902;
pub const CL_PLATFORM_VENDOR: ClPlatformInfo = 0x0903;
pub const CL_PLATFORM_VERSION: ClPlatformInfo = 0x0904;

pub const CL_DEVICE_NAME: ClDeviceInfo = 0x102B;
pub const CL_DEVICE_VENDOR: ClDeviceInfo = 0x102C;
pub const CL_DEVICE_VERSION: ClDeviceInfo = 0x102F;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE: ClDeviceInfo = 0x1004;

pub const CL_MEM_READ_WRITE: ClMemFlags = 1 << 0;
pub const CL_MEM_WRITE_ONLY: ClMemFlags = 1 << 1;
pub const CL_MEM_READ_ONLY: ClMemFlags = 1 << 2;
pub const CL_MEM_USE_HOST_PTR: ClMemFlags = 1 << 3;
pub const CL_MEM_ALLOC_HOST_PTR: ClMemFlags = 1 << 4;
pub const CL_MEM_COPY_HOST_PTR: ClMemFlags = 1 << 5;

pub const CL_BLOCKING: ClBool = 1;
pub const CL_NON_BLOCKING: ClBool = 0;
pub const CL_FALSE: ClBool = 0;
pub const CL_TRUE: ClBool = 1;

pub const CL_PROGRAM_BUILD_STATUS: ClProgramBuildInfo = 0x1181;
pub const CL_PROGRAM_BUILD_LOG: ClProgramBuildInfo = 0x1183;
pub const CL_BUILD_SUCCESS: ClInt = 0;

pub const CL_PROFILING_COMMAND_QUEUED: ClProfilingInfo = 0x1280;
pub const CL_PROFILING_COMMAND_SUBMIT: ClProfilingInfo = 0x1281;
pub const CL_PROFILING_COMMAND_START: ClProfilingInfo = 0x1282;
pub const CL_PROFILING_COMMAND_END: ClProfilingInfo = 0x1283;
pub const CL_QUEUE_PROFILING_ENABLE: ClCommandQueueProperties = 1 << 1;

#[allow(non_snake_case)]
pub struct OpenClFunctions {
    pub clGetPlatformIDs: unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt,
    pub clGetPlatformInfo:
        unsafe extern "C" fn(ClPlatformId, ClPlatformInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clGetDeviceIDs: unsafe extern "C" fn(
        ClPlatformId,
        ClDeviceType,
        ClUint,
        *mut ClDeviceId,
        *mut ClUint,
    ) -> ClInt,
    pub clGetDeviceInfo:
        unsafe extern "C" fn(ClDeviceId, ClDeviceInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clCreateContext: unsafe extern "C" fn(
        *const isize,
        ClUint,
        *const ClDeviceId,
        Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut ClInt,
    ) -> ClContext,
    pub clReleaseContext: unsafe extern "C" fn(ClContext) -> ClInt,
    pub clCreateCommandQueue: unsafe extern "C" fn(
        ClContext,
        ClDeviceId,
        ClCommandQueueProperties,
        *mut ClInt,
    ) -> ClCommandQueue,
    pub clReleaseCommandQueue: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    pub clCreateProgramWithSource: unsafe extern "C" fn(
        ClContext,
        ClUint,
        *const *const c_char,
        *const usize,
        *mut ClInt,
    ) -> ClProgram,
    pub clBuildProgram: unsafe extern "C" fn(
        ClProgram,
        ClUint,
        *const ClDeviceId,
        *const c_char,
        Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
        *mut c_void,
    ) -> ClInt,
    pub clGetProgramBuildInfo: unsafe extern "C" fn(
        ClProgram,
        ClDeviceId,
        ClProgramBuildInfo,
        usize,
        *mut c_void,
        *mut usize,
    ) -> ClInt,
    pub clCreateKernel: unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel,
    pub clReleaseKernel: unsafe extern "C" fn(ClKernel) -> ClInt,
    pub clReleaseProgram: unsafe extern "C" fn(ClProgram) -> ClInt,
    pub clSetKernelArg: unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt,
    pub clCreateBuffer:
        unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem,
    pub clReleaseMemObject: unsafe extern "C" fn(ClMem) -> ClInt,
    pub clEnqueueWriteBuffer: unsafe extern "C" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        usize,
        usize,
        *const c_void,
        ClUint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clEnqueueReadBuffer: unsafe extern "C" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        usize,
        usize,
        *mut c_void,
        ClUint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clEnqueueNDRangeKernel: unsafe extern "C" fn(
        ClCommandQueue,
        ClKernel,
        ClUint,
        *const usize,
        *const usize,
        *const usize,
        ClUint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clFinish: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    pub clFlush: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    pub clWaitForEvents: unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt,
    pub clReleaseEvent: unsafe extern "C" fn(ClEvent) -> ClInt,
    pub clGetEventProfilingInfo:
        unsafe extern "C" fn(ClEvent, ClProfilingInfo, usize, *mut c_void, *mut usize) -> ClInt,
}

pub struct OpenClContext {
    cl_lib: *mut c_void,
    pub platform: ClPlatformId,
    pub device: ClDeviceId,
    pub context: ClContext,
    pub queue: ClCommandQueue,
    pub platform_name: String,
    pub device_name: String,
    pub is_gpu: bool,
    pub max_work_group_size: usize,
    pub cl: OpenClFunctions,
    pub queue_lock: Mutex<()>,
}

unsafe impl Send for OpenClContext {}
unsafe impl Sync for OpenClContext {}

impl OpenClContext {
    /// Discovers and initializes an OpenCL context.
    pub fn new() -> Result<Arc<Self>> {
        const CANDIDATES: &[&CStr] = &[
            c"libOpenCL.so.1",
            c"libOpenCL.so",
            c"/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            c"/usr/lib/aarch64-linux-gnu/libOpenCL.so.1",
            c"/system/vendor/lib64/libOpenCL.so",
            c"/vendor/lib64/libOpenCL.so",
            c"/vendor/lib64/egl/libGLES_mali.so",
            c"/vendor/lib64/libPVROCL.so",
            c"/system/lib64/libOpenCL.so",
            c"/usr/lib/wsl/lib/libOpenCL.so",
        ];

        let mut cl_lib: *mut c_void = std::ptr::null_mut();
        for &c_name in CANDIDATES {
            let handle =
                unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL) };
            if !handle.is_null() {
                cl_lib = handle;
                log::debug!("Successfully loaded OpenCL library: {:?}", c_name);
                break;
            }
        }

        if cl_lib.is_null() {
            return Err(ScalixError::BackendUnavailable(
                crate::types::BackendType::OpenCL,
            ));
        }

        macro_rules! get_cl_fn {
            ($c_name:expr, $t:ty) => {{
                let ptr = unsafe { libc::dlsym(cl_lib, $c_name.as_ptr()) };
                if ptr.is_null() {
                    unsafe { libc::dlclose(cl_lib) };
                    return Err(ScalixError::ExecutionFailed(format!(
                        "Missing OpenCL symbol: {:?}",
                        $c_name
                    )));
                }
                unsafe { std::mem::transmute::<*mut c_void, $t>(ptr) }
            }};
        }

        let cl = OpenClFunctions {
            clGetPlatformIDs: get_cl_fn!(
                c"clGetPlatformIDs",
                unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt
            ),
            clGetPlatformInfo: get_cl_fn!(
                c"clGetPlatformInfo",
                unsafe extern "C" fn(
                    ClPlatformId,
                    ClPlatformInfo,
                    usize,
                    *mut c_void,
                    *mut usize,
                ) -> ClInt
            ),
            clGetDeviceIDs: get_cl_fn!(
                c"clGetDeviceIDs",
                unsafe extern "C" fn(
                    ClPlatformId,
                    ClDeviceType,
                    ClUint,
                    *mut ClDeviceId,
                    *mut ClUint,
                ) -> ClInt
            ),
            clGetDeviceInfo: get_cl_fn!(
                c"clGetDeviceInfo",
                unsafe extern "C" fn(
                    ClDeviceId,
                    ClDeviceInfo,
                    usize,
                    *mut c_void,
                    *mut usize,
                ) -> ClInt
            ),
            clCreateContext: get_cl_fn!(
                c"clCreateContext",
                unsafe extern "C" fn(
                    *const isize,
                    ClUint,
                    *const ClDeviceId,
                    Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>,
                    *mut c_void,
                    *mut ClInt,
                ) -> ClContext
            ),
            clReleaseContext: get_cl_fn!(
                c"clReleaseContext",
                unsafe extern "C" fn(ClContext) -> ClInt
            ),
            clCreateCommandQueue: get_cl_fn!(
                c"clCreateCommandQueue",
                unsafe extern "C" fn(
                    ClContext,
                    ClDeviceId,
                    ClCommandQueueProperties,
                    *mut ClInt,
                ) -> ClCommandQueue
            ),
            clReleaseCommandQueue: get_cl_fn!(
                c"clReleaseCommandQueue",
                unsafe extern "C" fn(ClCommandQueue) -> ClInt
            ),
            clCreateProgramWithSource: get_cl_fn!(
                c"clCreateProgramWithSource",
                unsafe extern "C" fn(
                    ClContext,
                    ClUint,
                    *const *const c_char,
                    *const usize,
                    *mut ClInt,
                ) -> ClProgram
            ),
            clBuildProgram: get_cl_fn!(
                c"clBuildProgram",
                unsafe extern "C" fn(
                    ClProgram,
                    ClUint,
                    *const ClDeviceId,
                    *const c_char,
                    Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
                    *mut c_void,
                ) -> ClInt
            ),
            clGetProgramBuildInfo: get_cl_fn!(
                c"clGetProgramBuildInfo",
                unsafe extern "C" fn(
                    ClProgram,
                    ClDeviceId,
                    ClProgramBuildInfo,
                    usize,
                    *mut c_void,
                    *mut usize,
                ) -> ClInt
            ),
            clCreateKernel: get_cl_fn!(
                c"clCreateKernel",
                unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel
            ),
            clReleaseKernel: get_cl_fn!(
                c"clReleaseKernel",
                unsafe extern "C" fn(ClKernel) -> ClInt
            ),
            clReleaseProgram: get_cl_fn!(
                c"clReleaseProgram",
                unsafe extern "C" fn(ClProgram) -> ClInt
            ),
            clSetKernelArg: get_cl_fn!(
                c"clSetKernelArg",
                unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt
            ),
            clCreateBuffer: get_cl_fn!(
                c"clCreateBuffer",
                unsafe extern "C" fn(
                    ClContext,
                    ClMemFlags,
                    usize,
                    *mut c_void,
                    *mut ClInt,
                ) -> ClMem
            ),
            clReleaseMemObject: get_cl_fn!(
                c"clReleaseMemObject",
                unsafe extern "C" fn(ClMem) -> ClInt
            ),
            clEnqueueWriteBuffer: get_cl_fn!(
                c"clEnqueueWriteBuffer",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClMem,
                    ClBool,
                    usize,
                    usize,
                    *const c_void,
                    ClUint,
                    *const ClEvent,
                    *mut ClEvent,
                ) -> ClInt
            ),
            clEnqueueReadBuffer: get_cl_fn!(
                c"clEnqueueReadBuffer",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClMem,
                    ClBool,
                    usize,
                    usize,
                    *mut c_void,
                    ClUint,
                    *const ClEvent,
                    *mut ClEvent,
                ) -> ClInt
            ),
            clEnqueueNDRangeKernel: get_cl_fn!(
                c"clEnqueueNDRangeKernel",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClKernel,
                    ClUint,
                    *const usize,
                    *const usize,
                    *const usize,
                    ClUint,
                    *const ClEvent,
                    *mut ClEvent,
                ) -> ClInt
            ),
            clFinish: get_cl_fn!(c"clFinish", unsafe extern "C" fn(ClCommandQueue) -> ClInt),
            clFlush: get_cl_fn!(c"clFlush", unsafe extern "C" fn(ClCommandQueue) -> ClInt),
            clWaitForEvents: get_cl_fn!(
                c"clWaitForEvents",
                unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt
            ),
            clReleaseEvent: get_cl_fn!(c"clReleaseEvent", unsafe extern "C" fn(ClEvent) -> ClInt),
            clGetEventProfilingInfo: get_cl_fn!(
                c"clGetEventProfilingInfo",
                unsafe extern "C" fn(
                    ClEvent,
                    ClProfilingInfo,
                    usize,
                    *mut c_void,
                    *mut usize,
                ) -> ClInt
            ),
        };

        // 1. Enumerate platforms
        let mut num_platforms: ClUint = 0;
        let err = unsafe { (cl.clGetPlatformIDs)(0, std::ptr::null_mut(), &mut num_platforms) };
        if err != CL_SUCCESS || num_platforms == 0 {
            unsafe { libc::dlclose(cl_lib) };
            return Err(ScalixError::BackendUnavailable(
                crate::types::BackendType::OpenCL,
            ));
        }

        let mut platforms: Vec<ClPlatformId> = vec![std::ptr::null_mut(); num_platforms as usize];
        unsafe {
            (cl.clGetPlatformIDs)(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut())
        };

        // 2. Discover best platform and device (prioritize GPU > CPU > Accelerator)
        let mut chosen_platform: Option<ClPlatformId> = None;
        let mut chosen_device: Option<ClDeviceId> = None;
        let mut is_gpu = false;

        for &platform in &platforms {
            let mut num_devices: ClUint = 0;
            // Try GPU first
            let res = unsafe {
                (cl.clGetDeviceIDs)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    0,
                    std::ptr::null_mut(),
                    &mut num_devices,
                )
            };
            if res == CL_SUCCESS && num_devices > 0 {
                let mut devices: Vec<ClDeviceId> = vec![std::ptr::null_mut(); num_devices as usize];
                unsafe {
                    (cl.clGetDeviceIDs)(
                        platform,
                        CL_DEVICE_TYPE_GPU,
                        num_devices,
                        devices.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                chosen_platform = Some(platform);
                chosen_device = Some(devices[0]);
                is_gpu = true;
                break;
            }
        }

        if chosen_device.is_none() {
            // Fallback to any device (CPU / Accelerator / Default)
            for &platform in &platforms {
                let mut num_devices: ClUint = 0;
                let res = unsafe {
                    (cl.clGetDeviceIDs)(
                        platform,
                        CL_DEVICE_TYPE_ALL,
                        0,
                        std::ptr::null_mut(),
                        &mut num_devices,
                    )
                };
                if res == CL_SUCCESS && num_devices > 0 {
                    let mut devices: Vec<ClDeviceId> =
                        vec![std::ptr::null_mut(); num_devices as usize];
                    unsafe {
                        (cl.clGetDeviceIDs)(
                            platform,
                            CL_DEVICE_TYPE_ALL,
                            num_devices,
                            devices.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    };
                    chosen_platform = Some(platform);
                    chosen_device = Some(devices[0]);
                    is_gpu = false;
                    break;
                }
            }
        }

        let (platform, device) = match (chosen_platform, chosen_device) {
            (Some(p), Some(d)) => (p, d),
            _ => {
                unsafe { libc::dlclose(cl_lib) };
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenCL,
                ));
            }
        };

        // Query platform & device info strings
        let query_string = |info_type: u32, is_platform: bool| -> String {
            let mut size: usize = 0;
            if is_platform {
                unsafe {
                    (cl.clGetPlatformInfo)(platform, info_type, 0, std::ptr::null_mut(), &mut size)
                };
            } else {
                unsafe {
                    (cl.clGetDeviceInfo)(device, info_type, 0, std::ptr::null_mut(), &mut size)
                };
            }
            if size == 0 {
                return String::new();
            }
            let mut buf = vec![0u8; size];
            if is_platform {
                unsafe {
                    (cl.clGetPlatformInfo)(
                        platform,
                        info_type,
                        size,
                        buf.as_mut_ptr() as *mut c_void,
                        std::ptr::null_mut(),
                    )
                };
            } else {
                unsafe {
                    (cl.clGetDeviceInfo)(
                        device,
                        info_type,
                        size,
                        buf.as_mut_ptr() as *mut c_void,
                        std::ptr::null_mut(),
                    )
                };
            }
            CStr::from_bytes_until_nul(&buf)
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
        };

        let platform_name = query_string(CL_PLATFORM_NAME, true);
        let device_name = query_string(CL_DEVICE_NAME, false);

        let mut max_work_group_size: usize = 256;
        unsafe {
            (cl.clGetDeviceInfo)(
                device,
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                &mut max_work_group_size as *mut usize as *mut c_void,
                std::ptr::null_mut(),
            )
        };

        log::debug!(
            "OpenCL context discovered: Platform='{}', Device='{}' (is_gpu={}, max_wg_size={})",
            platform_name,
            device_name,
            is_gpu,
            max_work_group_size
        );

        // 3. Create context
        let mut err: ClInt = 0;
        let context = unsafe {
            (cl.clCreateContext)(
                std::ptr::null(),
                1,
                &device,
                None,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS || context.is_null() {
            unsafe { libc::dlclose(cl_lib) };
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to create OpenCL context: err={}",
                err
            )));
        }

        // 4. Create command queue with profiling enabled
        let mut queue_props = CL_QUEUE_PROFILING_ENABLE;
        let mut queue =
            unsafe { (cl.clCreateCommandQueue)(context, device, queue_props, &mut err) };
        if err != CL_SUCCESS || queue.is_null() {
            // Fallback without profiling flags if driver restricts it
            queue_props = 0;
            queue = unsafe { (cl.clCreateCommandQueue)(context, device, queue_props, &mut err) };
        }
        if err != CL_SUCCESS || queue.is_null() {
            unsafe {
                (cl.clReleaseContext)(context);
                libc::dlclose(cl_lib);
            }
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to create OpenCL command queue: err={}",
                err
            )));
        }

        Ok(Arc::new(Self {
            cl_lib,
            platform,
            device,
            context,
            queue,
            platform_name,
            device_name,
            is_gpu,
            max_work_group_size,
            cl,
            queue_lock: Mutex::new(()),
        }))
    }
}

impl Drop for OpenClContext {
    fn drop(&mut self) {
        unsafe {
            if !self.queue.is_null() {
                (self.cl.clFinish)(self.queue);
                (self.cl.clReleaseCommandQueue)(self.queue);
            }
            if !self.context.is_null() {
                (self.cl.clReleaseContext)(self.context);
            }
            if !self.cl_lib.is_null() {
                libc::dlclose(self.cl_lib);
            }
        }
    }
}
