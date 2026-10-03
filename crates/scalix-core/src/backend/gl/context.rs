//! Headless EGL Context and OpenGL Function Loader

use std::ffi::{c_char, c_void, CString};
use std::sync::Mutex;

use crate::types::{Result, ScalixError};

pub type EGLDisplay = *mut c_void;
pub type EGLConfig = *mut c_void;
pub type EGLContext = *mut c_void;
pub type EGLSurface = *mut c_void;
pub type EGLint = i32;
pub type EGLBoolean = u32;

pub const EGL_FALSE: EGLBoolean = 0;
pub const EGL_TRUE: EGLBoolean = 1;
pub const EGL_DEFAULT_DISPLAY: EGLDisplay = std::ptr::null_mut();
pub const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
pub const EGL_NO_SURFACE: EGLSurface = std::ptr::null_mut();
pub const EGL_NO_DISPLAY: EGLDisplay = std::ptr::null_mut();

pub const EGL_SURFACE_TYPE: EGLint = 0x3033;
pub const EGL_PBUFFER_BIT: EGLint = 0x0001;
pub const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
pub const EGL_OPENGL_ES2_BIT: EGLint = 0x0004;
pub const EGL_OPENGL_ES3_BIT_KHR: EGLint = 0x0040;
pub const EGL_OPENGL_BIT: EGLint = 0x0008;
pub const EGL_RED_SIZE: EGLint = 0x3024;
pub const EGL_GREEN_SIZE: EGLint = 0x3023;
pub const EGL_BLUE_SIZE: EGLint = 0x3022;
pub const EGL_ALPHA_SIZE: EGLint = 0x3021;
pub const EGL_DEPTH_SIZE: EGLint = 0x3025;
pub const EGL_NONE: EGLint = 0x3038;
pub const EGL_CONTEXT_CLIENT_VERSION: EGLint = 0x3098;
pub const EGL_CONTEXT_MAJOR_VERSION: EGLint = 0x3098;
pub const EGL_CONTEXT_MINOR_VERSION: EGLint = 0x30FB;
pub const EGL_CONTEXT_OPENGL_PROFILE_MASK: EGLint = 0x30FD;
pub const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: EGLint = 0x00000001;

pub const EGL_OPENGL_API: u32 = 0x30A2;
pub const EGL_OPENGL_ES_API: u32 = 0x30A0;

pub const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
pub const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
pub const EGL_PLATFORM_GBM_KHR: u32 = 0x31D7;
pub const EGL_PLATFORM_GBM_MESA: u32 = 0x31D7;
pub const EGL_PLATFORM_WAYLAND_KHR: u32 = 0x31D8;
pub const EGL_PLATFORM_X11_KHR: u32 = 0x31D5;
pub const EGL_PLATFORM_ANDROID_KHR: u32 = 0x3141;

pub const EGL_WIDTH: EGLint = 0x3057;
pub const EGL_HEIGHT: EGLint = 0x3056;

pub fn format_egl_error(code: EGLint) -> &'static str {
    match code {
        0x3000 => "EGL_SUCCESS (0x3000)",
        0x3001 => "EGL_NOT_INITIALIZED (0x3001)",
        0x3002 => "EGL_BAD_ACCESS (0x3002)",
        0x3003 => "EGL_BAD_ALLOC (0x3003)",
        0x3004 => "EGL_BAD_ATTRIBUTE (0x3004)",
        0x3005 => "EGL_BAD_CONFIG (0x3005)",
        0x3006 => "EGL_BAD_CONTEXT (0x3006)",
        0x3007 => "EGL_BAD_CURRENT_SURFACE (0x3007)",
        0x3008 => "EGL_BAD_DISPLAY (0x3008)",
        0x3009 => "EGL_BAD_MATCH (0x3009)",
        0x300A => "EGL_BAD_NATIVE_PIXMAP (0x300A)",
        0x300B => "EGL_BAD_NATIVE_WINDOW (0x300B)",
        0x300C => "EGL_BAD_PARAMETER (0x300C)",
        0x300D => "EGL_BAD_SURFACE (0x300D)",
        0x300E => "EGL_CONTEXT_LOST (0x300E)",
        _ => "EGL_UNKNOWN_ERROR",
    }
}

pub fn format_gl_error(code: u32) -> &'static str {
    match code {
        0 => "GL_NO_ERROR (0)",
        0x0500 => "GL_INVALID_ENUM (0x0500)",
        0x0501 => "GL_INVALID_VALUE (0x0501)",
        0x0502 => "GL_INVALID_OPERATION (0x0502)",
        0x0503 => "GL_STACK_OVERFLOW (0x0503)",
        0x0504 => "GL_STACK_UNDERFLOW (0x0504)",
        0x0505 => "GL_OUT_OF_MEMORY (0x0505)",
        0x0506 => "GL_INVALID_FRAMEBUFFER_OPERATION (0x0506)",
        _ => "GL_UNKNOWN_ERROR",
    }
}

pub type EGLenum = u32;
pub type EGLAttrib = isize;
pub type EGLDeviceEXT = *mut c_void;

// GL Constants
pub const GL_TEXTURE_2D: u32 = 0x0DE1;
pub const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
pub const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
pub const GL_TEXTURE_WRAP_S: u32 = 0x2802;
pub const GL_TEXTURE_WRAP_T: u32 = 0x2803;
pub const GL_NEAREST: u32 = 0x2600;
pub const GL_LINEAR: u32 = 0x2601;
pub const GL_LINEAR_MIPMAP_LINEAR: u32 = 0x2703;
pub const GL_CLAMP_TO_EDGE: u32 = 0x812F;
pub const GL_RGBA: u32 = 0x1908;
pub const GL_RGB: u32 = 0x1907;
pub const GL_BGRA: u32 = 0x80E1;
pub const GL_BGR: u32 = 0x80E0;
pub const GL_RGBA8: u32 = 0x8058;
pub const GL_RGB8: u32 = 0x8051;
pub const GL_UNSIGNED_BYTE: u32 = 0x1401;
pub const GL_TEXTURE_BASE_LEVEL: u32 = 0x813C;
pub const GL_TEXTURE_MAX_LEVEL: u32 = 0x813D;

pub const GL_FRAMEBUFFER: u32 = 0x8D40;
pub const GL_READ_FRAMEBUFFER: u32 = 0x8CA8;
pub const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
pub const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
pub const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;

pub const GL_FLOAT: u32 = 0x1406;
pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_ARRAY_BUFFER: u32 = 0x8892;
pub const GL_STATIC_DRAW: u32 = 0x88E4;

pub const GL_VERTEX_SHADER: u32 = 0x8B31;
pub const GL_FRAGMENT_SHADER: u32 = 0x8B30;
pub const GL_COMPUTE_SHADER: u32 = 0x91B9;
pub const GL_COMPILE_STATUS: u32 = 0x8B81;
pub const GL_LINK_STATUS: u32 = 0x8B82;
pub const GL_INFO_LOG_LENGTH: u32 = 0x8B84;

pub const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
pub const GL_PACK_ALIGNMENT: u32 = 0x0D05;
pub const GL_TEXTURE0: u32 = 0x84C0;
pub const GL_WRITE_ONLY: u32 = 0x88B9;
pub const GL_READ_WRITE: u32 = 0x88BA;
pub const GL_SHADER_IMAGE_ACCESS_BARRIER_BIT: u32 = 0x00000020;
pub const GL_ALL_BARRIER_BITS: u32 = 0xFFFFFFFF;

pub const GL_PIXEL_UNPACK_BUFFER: u32 = 0x88EC;
pub const GL_PIXEL_PACK_BUFFER: u32 = 0x88EB;
pub const GL_STREAM_DRAW: u32 = 0x88E0;
pub const GL_STREAM_READ: u32 = 0x88E1;
pub const GL_DYNAMIC_DRAW: u32 = 0x88E8;
pub const GL_SYNC_GPU_COMMANDS_COMPLETE: u32 = 0x9117;
pub const GL_SYNC_FLUSH_COMMANDS_BIT: u32 = 0x00000001;
pub const GL_ALREADY_SIGNALED: u32 = 0x911A;
pub const GL_TIMEOUT_EXPIRED: u32 = 0x911B;
pub const GL_CONDITION_SATISFIED: u32 = 0x911C;
pub const GL_WAIT_FAILED: u32 = 0x911D;
pub const GL_TIMEOUT_IGNORED: u64 = 0xFFFFFFFFFFFFFFFF;

pub const GL_MAP_READ_BIT: u32 = 0x0001;
pub const GL_MAP_WRITE_BIT: u32 = 0x0002;
pub const GL_MAP_INVALIDATE_BUFFER_BIT: u32 = 0x0008;

pub type GLsync = *mut c_void;

#[allow(non_snake_case)]
pub struct GlFunctions {
    pub glGenTextures: unsafe extern "C" fn(i32, *mut u32),
    pub glBindTexture: unsafe extern "C" fn(u32, u32),
    pub glTexImage2D: unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    pub glTexSubImage2D:
        unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    pub glTexParameteri: unsafe extern "C" fn(u32, u32, i32),
    pub glDeleteTextures: unsafe extern "C" fn(i32, *const u32),
    pub glActiveTexture: unsafe extern "C" fn(u32),
    pub glGenerateMipmap: unsafe extern "C" fn(u32),

    pub glGenFramebuffers: unsafe extern "C" fn(i32, *mut u32),
    pub glBindFramebuffer: unsafe extern "C" fn(u32, u32),
    pub glFramebufferTexture2D: unsafe extern "C" fn(u32, u32, u32, u32, i32),
    pub glCheckFramebufferStatus: unsafe extern "C" fn(u32) -> u32,
    pub glDeleteFramebuffers: unsafe extern "C" fn(i32, *const u32),
    pub glBlitFramebuffer: unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32),
    pub glReadPixels: unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void),
    pub glViewport: unsafe extern "C" fn(i32, i32, i32, i32),
    pub glClear: unsafe extern "C" fn(u32),
    pub glClearColor: unsafe extern "C" fn(f32, f32, f32, f32),
    pub glPixelStorei: unsafe extern "C" fn(u32, i32),

    pub glCreateShader: unsafe extern "C" fn(u32) -> u32,
    pub glShaderSource: unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32),
    pub glCompileShader: unsafe extern "C" fn(u32),
    pub glGetShaderiv: unsafe extern "C" fn(u32, u32, *mut i32),
    pub glGetShaderInfoLog: unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char),
    pub glCreateProgram: unsafe extern "C" fn() -> u32,
    pub glAttachShader: unsafe extern "C" fn(u32, u32),
    pub glLinkProgram: unsafe extern "C" fn(u32),
    pub glGetProgramiv: unsafe extern "C" fn(u32, u32, *mut i32),
    pub glGetProgramInfoLog: unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char),
    pub glUseProgram: unsafe extern "C" fn(u32),
    pub glDeleteProgram: unsafe extern "C" fn(u32),
    pub glDeleteShader: unsafe extern "C" fn(u32),
    pub glGetUniformLocation: unsafe extern "C" fn(u32, *const c_char) -> i32,
    pub glUniform1i: unsafe extern "C" fn(i32, i32),
    pub glUniform1f: unsafe extern "C" fn(i32, f32),
    pub glUniform2f: unsafe extern "C" fn(i32, f32, f32),

    pub glGenVertexArrays: unsafe extern "C" fn(i32, *mut u32),
    pub glBindVertexArray: unsafe extern "C" fn(u32),
    pub glDeleteVertexArrays: unsafe extern "C" fn(i32, *const u32),
    pub glGenBuffers: unsafe extern "C" fn(i32, *mut u32),
    pub glBindBuffer: unsafe extern "C" fn(u32, u32),
    pub glBufferData: unsafe extern "C" fn(u32, isize, *const c_void, u32),
    pub glDeleteBuffers: unsafe extern "C" fn(i32, *const u32),
    pub glEnableVertexAttribArray: unsafe extern "C" fn(u32),
    pub glVertexAttribPointer: unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void),
    pub glDrawArrays: unsafe extern "C" fn(u32, i32, i32),

    pub glFinish: unsafe extern "C" fn(),
    pub glFlush: unsafe extern "C" fn(),
    pub glGetString: unsafe extern "C" fn(u32) -> *const c_char,

    // Optional Compute
    pub glDispatchCompute: Option<unsafe extern "C" fn(u32, u32, u32)>,
    pub glBindImageTexture: Option<unsafe extern "C" fn(u32, u32, i32, u8, i32, u32, u32)>,
    pub glMemoryBarrier: Option<unsafe extern "C" fn(u32)>,

    // Hardware Synchronization Fences (GL 3.2+ / GLES 3.0+)
    pub glFenceSync: Option<unsafe extern "C" fn(u32, u32) -> GLsync>,
    pub glClientWaitSync: Option<unsafe extern "C" fn(GLsync, u32, u64) -> u32>,
    pub glWaitSync: Option<unsafe extern "C" fn(GLsync, u32, u64)>,
    pub glDeleteSync: Option<unsafe extern "C" fn(GLsync)>,

    // Buffer Mapping
    pub glMapBufferRange: Option<unsafe extern "C" fn(u32, isize, isize, u32) -> *mut c_void>,
    pub glUnmapBuffer: Option<unsafe extern "C" fn(u32) -> u8>,

    // Error Reporting
    pub glGetError: Option<unsafe extern "C" fn() -> u32>,
}

pub type EglMakeCurrentFn =
    unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;
pub type EglDestroyContextFn = unsafe extern "C" fn(EGLDisplay, EGLContext) -> EGLBoolean;
pub type EglCreatePbufferSurfaceFn =
    unsafe extern "C" fn(EGLDisplay, EGLConfig, *const EGLint) -> EGLSurface;
pub type EglDestroySurfaceFn = unsafe extern "C" fn(EGLDisplay, EGLSurface) -> EGLBoolean;
pub type EglGetErrorFn = unsafe extern "C" fn() -> EGLint;
pub type EglTerminateFn = unsafe extern "C" fn(EGLDisplay) -> EGLBoolean;
pub type GbmDestroyDeviceFn = unsafe extern "C" fn(*mut c_void);

pub struct EglContext {
    egl_lib: *mut c_void,
    gl_lib: *mut c_void,
    gbm_lib: *mut c_void,
    gbm_dev: *mut c_void,
    gbm_fd: i32,
    gbm_destroy_device: Option<GbmDestroyDeviceFn>,
    display: EGLDisplay,
    context: EGLContext,
    pub surface: EGLSurface,
    egl_make_current: EglMakeCurrentFn,
    egl_destroy_context: Option<EglDestroyContextFn>,
    egl_destroy_surface: Option<EglDestroySurfaceFn>,
    egl_get_error: Option<EglGetErrorFn>,
    egl_terminate: Option<EglTerminateFn>,
    pub gl: GlFunctions,
    pub is_compute_supported: bool,
    pub is_gles: bool,
    pub gl_version: String,
    _lock: Mutex<()>,
}

pub struct EglContextGuard<'a> {
    ctx: &'a EglContext,
    #[allow(dead_code)]
    _lock_guard: std::sync::MutexGuard<'a, ()>,
}

impl<'a> Drop for EglContextGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.egl_make_current)(
                self.ctx.display,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                EGL_NO_CONTEXT,
            );
        }
    }
}

unsafe impl Send for EglContext {}
unsafe impl Sync for EglContext {}

impl EglContext {
    pub fn new() -> Result<Self> {
        unsafe {
            let egl_lib = libc::dlopen(
                c"libEGL.so.1".as_ptr() as *const c_char,
                libc::RTLD_LAZY | libc::RTLD_GLOBAL,
            );
            let egl_lib = if egl_lib.is_null() {
                libc::dlopen(
                    c"libEGL.so".as_ptr() as *const c_char,
                    libc::RTLD_LAZY | libc::RTLD_GLOBAL,
                )
            } else {
                egl_lib
            };

            if egl_lib.is_null() {
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }

            let gl_lib = libc::dlopen(
                c"libOpenGL.so.0".as_ptr() as *const c_char,
                libc::RTLD_LAZY | libc::RTLD_GLOBAL,
            );
            let gl_lib = if gl_lib.is_null() {
                libc::dlopen(
                    c"libGL.so.1".as_ptr() as *const c_char,
                    libc::RTLD_LAZY | libc::RTLD_GLOBAL,
                )
            } else {
                gl_lib
            };
            let gl_lib = if gl_lib.is_null() {
                libc::dlopen(
                    c"libGLESv2.so.2".as_ptr() as *const c_char,
                    libc::RTLD_LAZY | libc::RTLD_GLOBAL,
                )
            } else {
                gl_lib
            };

            type EglGetProcAddressFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
            let egl_get_proc_address_ptr =
                libc::dlsym(egl_lib, c"eglGetProcAddress".as_ptr() as *const c_char);
            let egl_get_proc_address: Option<EglGetProcAddressFn> =
                if !egl_get_proc_address_ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglGetProcAddressFn>(
                        egl_get_proc_address_ptr,
                    ))
                } else {
                    None
                };

            // Core EGL 1.0-1.4 symbols MUST be loaded via dlsym to avoid libglvnd dummy GL dispatch stubs
            let load_egl_core_symbol = |name: &str| -> *mut c_void {
                let c_name = CString::new(name).unwrap();
                libc::dlsym(egl_lib, c_name.as_ptr())
            };

            // EGL extension symbols (or EGL 1.5) can be resolved via dlsym or eglGetProcAddress
            let load_egl_ext_symbol = |name: &str| -> *mut c_void {
                let c_name = CString::new(name).unwrap();
                let mut ptr = libc::dlsym(egl_lib, c_name.as_ptr());
                if ptr.is_null() {
                    if let Some(egpa) = egl_get_proc_address {
                        ptr = egpa(c_name.as_ptr());
                    }
                }
                ptr
            };

            // OpenGL client API symbols are resolved via eglGetProcAddress or libOpenGL/libGL/libGLESv2 dlsym
            let load_gl_symbol = |name: &str| -> *mut c_void {
                let c_name = CString::new(name).unwrap();
                let mut ptr = if let Some(egpa) = egl_get_proc_address {
                    egpa(c_name.as_ptr())
                } else {
                    std::ptr::null_mut()
                };
                if ptr.is_null() && !gl_lib.is_null() {
                    ptr = libc::dlsym(gl_lib, c_name.as_ptr());
                }
                if ptr.is_null() {
                    ptr = libc::dlsym(egl_lib, c_name.as_ptr());
                }
                ptr
            };

            type EglGetDisplayFn = unsafe extern "C" fn(EGLDisplay) -> EGLDisplay;
            type EglGetPlatformDisplayFn =
                unsafe extern "C" fn(EGLenum, *mut c_void, *const EGLAttrib) -> EGLDisplay;
            type EglGetPlatformDisplayExtFn =
                unsafe extern "C" fn(EGLenum, *mut c_void, *const EGLint) -> EGLDisplay;
            type EglQueryDevicesExtFn =
                unsafe extern "C" fn(EGLint, *mut EGLDeviceEXT, *mut EGLint) -> EGLBoolean;
            type EglInitializeFn =
                unsafe extern "C" fn(EGLDisplay, *mut EGLint, *mut EGLint) -> EGLBoolean;
            type EglBindApiFn = unsafe extern "C" fn(u32) -> EGLBoolean;
            type EglChooseConfigFn = unsafe extern "C" fn(
                EGLDisplay,
                *const EGLint,
                *mut EGLConfig,
                EGLint,
                *mut EGLint,
            ) -> EGLBoolean;
            type EglCreateContextFn = unsafe extern "C" fn(
                EGLDisplay,
                EGLConfig,
                EGLContext,
                *const EGLint,
            ) -> EGLContext;
            type EglMakeCurrentFn =
                unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;

            let egl_get_display: Option<EglGetDisplayFn> = {
                let ptr = load_egl_core_symbol("eglGetDisplay");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglGetDisplayFn>(ptr))
                } else {
                    None
                }
            };
            let egl_get_platform_display: Option<EglGetPlatformDisplayFn> = {
                let ptr = load_egl_ext_symbol("eglGetPlatformDisplay");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglGetPlatformDisplayFn>(
                        ptr,
                    ))
                } else {
                    None
                }
            };
            let egl_get_platform_display_ext: Option<EglGetPlatformDisplayExtFn> = {
                let ptr = load_egl_ext_symbol("eglGetPlatformDisplayEXT");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglGetPlatformDisplayExtFn>(ptr))
                } else {
                    None
                }
            };
            let egl_query_devices_ext: Option<EglQueryDevicesExtFn> = {
                let ptr = load_egl_ext_symbol("eglQueryDevicesEXT");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglQueryDevicesExtFn>(
                        ptr,
                    ))
                } else {
                    None
                }
            };

            let egl_init_ptr = load_egl_core_symbol("eglInitialize");
            if egl_init_ptr.is_null() {
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }
            let egl_initialize: EglInitializeFn =
                std::mem::transmute::<*mut c_void, EglInitializeFn>(egl_init_ptr);

            let egl_bind_api_ptr = load_egl_core_symbol("eglBindAPI");
            if egl_bind_api_ptr.is_null() {
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }
            let egl_bind_api: EglBindApiFn =
                std::mem::transmute::<*mut c_void, EglBindApiFn>(egl_bind_api_ptr);

            let egl_choose_config_ptr = load_egl_core_symbol("eglChooseConfig");
            if egl_choose_config_ptr.is_null() {
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }
            let egl_choose_config: EglChooseConfigFn =
                std::mem::transmute::<*mut c_void, EglChooseConfigFn>(egl_choose_config_ptr);

            let egl_create_context_ptr = load_egl_core_symbol("eglCreateContext");
            if egl_create_context_ptr.is_null() {
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }
            let egl_create_context: EglCreateContextFn =
                std::mem::transmute::<*mut c_void, EglCreateContextFn>(egl_create_context_ptr);

            let egl_make_current_ptr = load_egl_core_symbol("eglMakeCurrent");
            if egl_make_current_ptr.is_null() {
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }
            let egl_make_current: EglMakeCurrentFn =
                std::mem::transmute::<*mut c_void, EglMakeCurrentFn>(egl_make_current_ptr);

            let egl_destroy_context: Option<EglDestroyContextFn> = {
                let ptr = load_egl_core_symbol("eglDestroyContext");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglDestroyContextFn>(ptr))
                } else {
                    None
                }
            };
            let egl_create_pbuffer_surface: Option<EglCreatePbufferSurfaceFn> = {
                let ptr = load_egl_core_symbol("eglCreatePbufferSurface");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglCreatePbufferSurfaceFn>(ptr))
                } else {
                    None
                }
            };
            let egl_destroy_surface: Option<EglDestroySurfaceFn> = {
                let ptr = load_egl_core_symbol("eglDestroySurface");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglDestroySurfaceFn>(ptr))
                } else {
                    None
                }
            };
            let egl_get_error: Option<EglGetErrorFn> = {
                let ptr = load_egl_core_symbol("eglGetError");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglGetErrorFn>(ptr))
                } else {
                    None
                }
            };
            let egl_terminate: Option<EglTerminateFn> = {
                let ptr = load_egl_core_symbol("eglTerminate");
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, EglTerminateFn>(ptr))
                } else {
                    None
                }
            };

            let get_platform_display =
                |platform: EGLenum, native_display: *mut c_void| -> EGLDisplay {
                    if let Some(f) = egl_get_platform_display_ext {
                        let d = f(platform, native_display, std::ptr::null());
                        if d != EGL_NO_DISPLAY {
                            return d;
                        }
                    }
                    if let Some(f) = egl_get_platform_display {
                        let d = f(platform, native_display, std::ptr::null());
                        if d != EGL_NO_DISPLAY {
                            return d;
                        }
                    }
                    EGL_NO_DISPLAY
                };

            type GbmHandle = (*mut c_void, i32);
            type CandidateDisplay = (EGLDisplay, &'static str, Option<GbmHandle>);

            let mut candidate_displays: Vec<CandidateDisplay> = Vec::new();

            // Strategy 1: EGL Device EXT (Native Discrete GPUs / NVIDIA / AMD / Intel directly)
            if let Some(query_devices) = egl_query_devices_ext {
                let mut devices: [EGLDeviceEXT; 16] = [std::ptr::null_mut(); 16];
                let mut num_devices: EGLint = 0;
                if query_devices(16, devices.as_mut_ptr(), &mut num_devices) == EGL_TRUE
                    && num_devices > 0
                {
                    for &device in devices.iter().take(num_devices as usize) {
                        let dev_display = get_platform_display(EGL_PLATFORM_DEVICE_EXT, device);
                        if dev_display != EGL_NO_DISPLAY {
                            candidate_displays.push((dev_display, "EGL_PLATFORM_DEVICE_EXT", None));
                        }
                    }
                }
            }

            // Strategy 2: GBM Platform (Headless DRM / Linux Render Nodes via libgbm)
            let gbm_lib = libc::dlopen(
                c"libgbm.so.1".as_ptr() as *const c_char,
                libc::RTLD_LAZY | libc::RTLD_LOCAL,
            );
            let gbm_lib = if gbm_lib.is_null() {
                libc::dlopen(
                    c"libgbm.so".as_ptr() as *const c_char,
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            } else {
                gbm_lib
            };

            let gbm_destroy_device: Option<GbmDestroyDeviceFn> = if !gbm_lib.is_null() {
                let ptr = libc::dlsym(gbm_lib, c"gbm_device_destroy".as_ptr() as *const c_char);
                if !ptr.is_null() {
                    Some(std::mem::transmute::<*mut c_void, GbmDestroyDeviceFn>(ptr))
                } else {
                    None
                }
            } else {
                None
            };

            if !gbm_lib.is_null() {
                type GbmCreateDeviceFn = unsafe extern "C" fn(i32) -> *mut c_void;
                let create_ptr =
                    libc::dlsym(gbm_lib, c"gbm_create_device".as_ptr() as *const c_char);
                if !create_ptr.is_null() {
                    let gbm_create_device: GbmCreateDeviceFn =
                        std::mem::transmute::<*mut c_void, GbmCreateDeviceFn>(create_ptr);
                    for &path in crate::dma::linux_drm::DRM_CANDIDATE_PATHS {
                        let c_path = CString::new(path).unwrap();
                        let fd = libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
                        if fd >= 0 {
                            let gbm_dev = gbm_create_device(fd);
                            if !gbm_dev.is_null() {
                                let gbm_display =
                                    get_platform_display(EGL_PLATFORM_GBM_KHR, gbm_dev);
                                if gbm_display != EGL_NO_DISPLAY {
                                    candidate_displays.push((
                                        gbm_display,
                                        "EGL_PLATFORM_GBM_KHR",
                                        Some((gbm_dev, fd)),
                                    ));
                                } else {
                                    if let Some(destroy) = gbm_destroy_device {
                                        destroy(gbm_dev);
                                    }
                                    libc::close(fd);
                                }
                            } else {
                                libc::close(fd);
                            }
                        }
                    }
                }
            }

            // Strategy 3: Surfaceless MESA (Headless Fallback / CI / VM Software Rasterizers)
            let surfaceless_display =
                get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY);
            if surfaceless_display != EGL_NO_DISPLAY {
                candidate_displays.push((
                    surfaceless_display,
                    "EGL_PLATFORM_SURFACELESS_MESA",
                    None,
                ));
            }

            // Strategy 4: Standard default display fallback (X11 / Wayland / Android / Windows)
            if let Some(egl_get_display) = egl_get_display {
                let def_display = egl_get_display(EGL_DEFAULT_DISPLAY);
                if def_display != EGL_NO_DISPLAY {
                    candidate_displays.push((def_display, "eglGetDisplay default", None));
                }
            }

            let mut display = EGL_NO_DISPLAY;
            let mut context = EGL_NO_CONTEXT;
            let mut surface = EGL_NO_SURFACE;
            let mut bound_major: EGLint = 0;
            let mut bound_minor: EGLint = 0;
            let mut selected_gbm: Option<(*mut c_void, i32)> = None;

            for (cand_display, cand_desc, cand_gbm) in candidate_displays.iter().copied() {
                let mut major: EGLint = 0;
                let mut minor: EGLint = 0;
                if egl_initialize(cand_display, &mut major, &mut minor) == EGL_FALSE {
                    continue;
                }

                let _ = egl_bind_api(EGL_OPENGL_API);

                let config_attribs = [
                    EGL_SURFACE_TYPE,
                    EGL_PBUFFER_BIT,
                    EGL_RED_SIZE,
                    8,
                    EGL_GREEN_SIZE,
                    8,
                    EGL_BLUE_SIZE,
                    8,
                    EGL_ALPHA_SIZE,
                    8,
                    EGL_NONE,
                ];

                let mut config: EGLConfig = std::ptr::null_mut();
                let mut num_configs: EGLint = 0;
                if egl_choose_config(
                    cand_display,
                    config_attribs.as_ptr(),
                    &mut config,
                    1,
                    &mut num_configs,
                ) == EGL_FALSE
                    || num_configs < 1
                {
                    let minimal_attribs = [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE];
                    let _ = egl_choose_config(
                        cand_display,
                        minimal_attribs.as_ptr(),
                        &mut config,
                        1,
                        &mut num_configs,
                    );
                }

                if num_configs < 1 {
                    let surfaceless_attribs = [EGL_NONE];
                    let _ = egl_choose_config(
                        cand_display,
                        surfaceless_attribs.as_ptr(),
                        &mut config,
                        1,
                        &mut num_configs,
                    );
                }

                if num_configs < 1 {
                    if let Some(f) = egl_terminate {
                        let _ = f(cand_display);
                    }
                    continue;
                }

                let context_attribs_gl43 = [
                    EGL_CONTEXT_MAJOR_VERSION,
                    4,
                    EGL_CONTEXT_MINOR_VERSION,
                    3,
                    EGL_CONTEXT_OPENGL_PROFILE_MASK,
                    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                    EGL_NONE,
                ];
                let context_attribs_gl33 = [
                    EGL_CONTEXT_MAJOR_VERSION,
                    3,
                    EGL_CONTEXT_MINOR_VERSION,
                    3,
                    EGL_CONTEXT_OPENGL_PROFILE_MASK,
                    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                    EGL_NONE,
                ];

                let mut cand_context = egl_create_context(
                    cand_display,
                    config,
                    EGL_NO_CONTEXT,
                    context_attribs_gl43.as_ptr(),
                );
                if cand_context == EGL_NO_CONTEXT {
                    cand_context = egl_create_context(
                        cand_display,
                        config,
                        EGL_NO_CONTEXT,
                        context_attribs_gl33.as_ptr(),
                    );
                }
                if cand_context == EGL_NO_CONTEXT {
                    let _ = egl_bind_api(EGL_OPENGL_ES_API);
                    let gles_attribs_31 = [
                        EGL_CONTEXT_CLIENT_VERSION,
                        3,
                        EGL_CONTEXT_MINOR_VERSION,
                        1,
                        EGL_NONE,
                    ];
                    cand_context = egl_create_context(
                        cand_display,
                        config,
                        EGL_NO_CONTEXT,
                        gles_attribs_31.as_ptr(),
                    );
                    if cand_context == EGL_NO_CONTEXT {
                        let gles_attribs_30 = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
                        cand_context = egl_create_context(
                            cand_display,
                            config,
                            EGL_NO_CONTEXT,
                            gles_attribs_30.as_ptr(),
                        );
                    }
                    if cand_context == EGL_NO_CONTEXT {
                        cand_context = egl_create_context(
                            cand_display,
                            config,
                            EGL_NO_CONTEXT,
                            std::ptr::null(),
                        );
                    }
                }

                if cand_context == EGL_NO_CONTEXT {
                    if let Some(f) = egl_terminate {
                        let _ = f(cand_display);
                    }
                    continue;
                }

                // Try binding: 1. Surfaceless (EGL_NO_SURFACE), 2. 1x1 Offscreen PBuffer
                if egl_make_current(cand_display, EGL_NO_SURFACE, EGL_NO_SURFACE, cand_context)
                    == EGL_TRUE
                {
                    display = cand_display;
                    context = cand_context;
                    surface = EGL_NO_SURFACE;
                    bound_major = major;
                    bound_minor = minor;
                    selected_gbm = cand_gbm;
                    log::debug!(
                        "Successfully bound headless EGL display via {} (surfaceless)",
                        cand_desc
                    );
                    break;
                }

                let pbuffer_attribs = [EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE];
                let pbuffer = egl_create_pbuffer_surface
                    .map(|create| create(cand_display, config, pbuffer_attribs.as_ptr()))
                    .unwrap_or(EGL_NO_SURFACE);

                if pbuffer != EGL_NO_SURFACE
                    && egl_make_current(cand_display, pbuffer, pbuffer, cand_context) == EGL_TRUE
                {
                    display = cand_display;
                    context = cand_context;
                    surface = pbuffer;
                    bound_major = major;
                    bound_minor = minor;
                    selected_gbm = cand_gbm;
                    log::debug!(
                        "Successfully bound headless EGL display via {} (1x1 PBuffer)",
                        cand_desc
                    );
                    break;
                }

                // Cleanup failed candidate
                if pbuffer != EGL_NO_SURFACE {
                    if let Some(f) = egl_destroy_surface {
                        let _ = f(cand_display, pbuffer);
                    }
                }
                if let Some(f) = egl_destroy_context {
                    let _ = f(cand_display, cand_context);
                }
                if let Some(f) = egl_terminate {
                    let _ = f(cand_display);
                }
            }

            // Immediately destroy all unselected candidate GBM devices and close their file descriptors
            for (_, _, cand_gbm) in candidate_displays {
                if let Some((g_dev, g_fd)) = cand_gbm {
                    if Some((g_dev, g_fd)) != selected_gbm {
                        if let Some(destroy) = gbm_destroy_device {
                            destroy(g_dev);
                        }
                        libc::close(g_fd);
                    }
                }
            }

            if display == EGL_NO_DISPLAY || context == EGL_NO_CONTEXT {
                if let Some((g_dev, g_fd)) = selected_gbm {
                    if let Some(destroy) = gbm_destroy_device {
                        destroy(g_dev);
                    }
                    libc::close(g_fd);
                }
                if !gbm_lib.is_null() {
                    libc::dlclose(gbm_lib);
                }
                if !gl_lib.is_null() {
                    libc::dlclose(gl_lib);
                }
                libc::dlclose(egl_lib);
                return Err(ScalixError::BackendUnavailable(
                    crate::types::BackendType::OpenGL,
                ));
            }

            log::debug!("Initialized headless EGL version {bound_major}.{bound_minor}");

            macro_rules! get_gl_fn {
                ($name:expr, $t:ty) => {{
                    let ptr = load_gl_symbol($name);
                    if ptr.is_null() {
                        return Err(ScalixError::ExecutionFailed(format!(
                            "Missing GL symbol: {}",
                            $name
                        )));
                    }
                    std::mem::transmute::<*mut c_void, $t>(ptr)
                }};
            }

            macro_rules! get_optional_gl_fn {
                ($name:expr, $t:ty) => {{
                    let ptr = load_gl_symbol($name);
                    if ptr.is_null() {
                        None
                    } else {
                        Some(std::mem::transmute::<*mut c_void, $t>(ptr))
                    }
                }};
            }

            let gl = GlFunctions {
                glGenTextures: get_gl_fn!("glGenTextures", unsafe extern "C" fn(i32, *mut u32)),
                glBindTexture: get_gl_fn!("glBindTexture", unsafe extern "C" fn(u32, u32)),
                glTexImage2D: get_gl_fn!(
                    "glTexImage2D",
                    unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
                ),
                glTexSubImage2D: get_gl_fn!(
                    "glTexSubImage2D",
                    unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
                ),
                glTexParameteri: get_gl_fn!("glTexParameteri", unsafe extern "C" fn(u32, u32, i32)),
                glDeleteTextures: get_gl_fn!(
                    "glDeleteTextures",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glActiveTexture: get_gl_fn!("glActiveTexture", unsafe extern "C" fn(u32)),
                glGenerateMipmap: get_gl_fn!("glGenerateMipmap", unsafe extern "C" fn(u32)),

                glGenFramebuffers: get_gl_fn!(
                    "glGenFramebuffers",
                    unsafe extern "C" fn(i32, *mut u32)
                ),
                glBindFramebuffer: get_gl_fn!("glBindFramebuffer", unsafe extern "C" fn(u32, u32)),
                glFramebufferTexture2D: get_gl_fn!(
                    "glFramebufferTexture2D",
                    unsafe extern "C" fn(u32, u32, u32, u32, i32)
                ),
                glCheckFramebufferStatus: get_gl_fn!(
                    "glCheckFramebufferStatus",
                    unsafe extern "C" fn(u32) -> u32
                ),
                glDeleteFramebuffers: get_gl_fn!(
                    "glDeleteFramebuffers",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glBlitFramebuffer: get_gl_fn!(
                    "glBlitFramebuffer",
                    unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32)
                ),
                glReadPixels: get_gl_fn!(
                    "glReadPixels",
                    unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void)
                ),
                glViewport: get_gl_fn!("glViewport", unsafe extern "C" fn(i32, i32, i32, i32)),
                glClear: get_gl_fn!("glClear", unsafe extern "C" fn(u32)),
                glClearColor: get_gl_fn!("glClearColor", unsafe extern "C" fn(f32, f32, f32, f32)),
                glPixelStorei: get_gl_fn!("glPixelStorei", unsafe extern "C" fn(u32, i32)),

                glCreateShader: get_gl_fn!("glCreateShader", unsafe extern "C" fn(u32) -> u32),
                glShaderSource: get_gl_fn!(
                    "glShaderSource",
                    unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32)
                ),
                glCompileShader: get_gl_fn!("glCompileShader", unsafe extern "C" fn(u32)),
                glGetShaderiv: get_gl_fn!(
                    "glGetShaderiv",
                    unsafe extern "C" fn(u32, u32, *mut i32)
                ),
                glGetShaderInfoLog: get_gl_fn!(
                    "glGetShaderInfoLog",
                    unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)
                ),
                glCreateProgram: get_gl_fn!("glCreateProgram", unsafe extern "C" fn() -> u32),
                glAttachShader: get_gl_fn!("glAttachShader", unsafe extern "C" fn(u32, u32)),
                glLinkProgram: get_gl_fn!("glLinkProgram", unsafe extern "C" fn(u32)),
                glGetProgramiv: get_gl_fn!(
                    "glGetProgramiv",
                    unsafe extern "C" fn(u32, u32, *mut i32)
                ),
                glGetProgramInfoLog: get_gl_fn!(
                    "glGetProgramInfoLog",
                    unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)
                ),
                glUseProgram: get_gl_fn!("glUseProgram", unsafe extern "C" fn(u32)),
                glDeleteProgram: get_gl_fn!("glDeleteProgram", unsafe extern "C" fn(u32)),
                glDeleteShader: get_gl_fn!("glDeleteShader", unsafe extern "C" fn(u32)),
                glGetUniformLocation: get_gl_fn!(
                    "glGetUniformLocation",
                    unsafe extern "C" fn(u32, *const c_char) -> i32
                ),
                glUniform1i: get_gl_fn!("glUniform1i", unsafe extern "C" fn(i32, i32)),
                glUniform1f: get_gl_fn!("glUniform1f", unsafe extern "C" fn(i32, f32)),
                glUniform2f: get_gl_fn!("glUniform2f", unsafe extern "C" fn(i32, f32, f32)),

                glGenVertexArrays: get_gl_fn!(
                    "glGenVertexArrays",
                    unsafe extern "C" fn(i32, *mut u32)
                ),
                glBindVertexArray: get_gl_fn!("glBindVertexArray", unsafe extern "C" fn(u32)),
                glDeleteVertexArrays: get_gl_fn!(
                    "glDeleteVertexArrays",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glGenBuffers: get_gl_fn!("glGenBuffers", unsafe extern "C" fn(i32, *mut u32)),
                glBindBuffer: get_gl_fn!("glBindBuffer", unsafe extern "C" fn(u32, u32)),
                glBufferData: get_gl_fn!(
                    "glBufferData",
                    unsafe extern "C" fn(u32, isize, *const c_void, u32)
                ),
                glDeleteBuffers: get_gl_fn!(
                    "glDeleteBuffers",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glEnableVertexAttribArray: get_gl_fn!(
                    "glEnableVertexAttribArray",
                    unsafe extern "C" fn(u32)
                ),
                glVertexAttribPointer: get_gl_fn!(
                    "glVertexAttribPointer",
                    unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void)
                ),
                glDrawArrays: get_gl_fn!("glDrawArrays", unsafe extern "C" fn(u32, i32, i32)),

                glFinish: get_gl_fn!("glFinish", unsafe extern "C" fn()),
                glFlush: get_gl_fn!("glFlush", unsafe extern "C" fn()),
                glGetString: get_gl_fn!("glGetString", unsafe extern "C" fn(u32) -> *const c_char),

                glDispatchCompute: get_optional_gl_fn!(
                    "glDispatchCompute",
                    unsafe extern "C" fn(u32, u32, u32)
                ),
                glBindImageTexture: get_optional_gl_fn!(
                    "glBindImageTexture",
                    unsafe extern "C" fn(u32, u32, i32, u8, i32, u32, u32)
                ),
                glMemoryBarrier: get_optional_gl_fn!("glMemoryBarrier", unsafe extern "C" fn(u32)),

                glFenceSync: get_optional_gl_fn!(
                    "glFenceSync",
                    unsafe extern "C" fn(u32, u32) -> GLsync
                ),
                glClientWaitSync: get_optional_gl_fn!(
                    "glClientWaitSync",
                    unsafe extern "C" fn(GLsync, u32, u64) -> u32
                ),
                glWaitSync: get_optional_gl_fn!(
                    "glWaitSync",
                    unsafe extern "C" fn(GLsync, u32, u64)
                ),
                glDeleteSync: get_optional_gl_fn!("glDeleteSync", unsafe extern "C" fn(GLsync)),

                glMapBufferRange: get_optional_gl_fn!(
                    "glMapBufferRange",
                    unsafe extern "C" fn(u32, isize, isize, u32) -> *mut c_void
                ),
                glUnmapBuffer: get_optional_gl_fn!(
                    "glUnmapBuffer",
                    unsafe extern "C" fn(u32) -> u8
                ),
                glGetError: get_optional_gl_fn!("glGetError", unsafe extern "C" fn() -> u32),
            };

            let is_compute_supported =
                gl.glDispatchCompute.is_some() && gl.glBindImageTexture.is_some();

            let version_ptr = (gl.glGetString)(0x1F02); // GL_VERSION = 0x1F02
            let version_str = if !version_ptr.is_null() {
                std::ffi::CStr::from_ptr(version_ptr)
                    .to_string_lossy()
                    .to_string()
            } else {
                String::new()
            };
            let is_gles = version_str.contains("OpenGL ES") || version_str.contains("ES");

            log::debug!(
                "EGL/GL Context Initialized: Version '{}', is_gles = {}, compute = {}, offscreen_pbuffer = {}",
                version_str,
                is_gles,
                is_compute_supported,
                surface != EGL_NO_SURFACE
            );

            // Unbind from creation thread so worker threads can bind it
            let _ = egl_make_current(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

            let (gbm_dev_ptr, gbm_fd_val) = selected_gbm.unwrap_or((std::ptr::null_mut(), -1));
            let final_gbm_lib = if gbm_dev_ptr.is_null() {
                if !gbm_lib.is_null() {
                    libc::dlclose(gbm_lib);
                }
                std::ptr::null_mut()
            } else {
                gbm_lib
            };

            Ok(Self {
                egl_lib,
                gl_lib,
                gbm_lib: final_gbm_lib,
                gbm_dev: gbm_dev_ptr,
                gbm_fd: gbm_fd_val,
                gbm_destroy_device,
                display,
                context,
                surface,
                egl_make_current,
                egl_destroy_context,
                egl_destroy_surface,
                egl_get_error,
                egl_terminate,
                gl,
                is_compute_supported,
                is_gles,
                gl_version: version_str,
                _lock: Mutex::new(()),
            })
        }
    }

    pub fn shader_header(&self) -> &'static str {
        if self.is_gles {
            "#version 300 es\nprecision highp float;\n"
        } else {
            "#version 330 core\n"
        }
    }

    pub fn compute_shader_header(&self) -> &'static str {
        if self.is_gles {
            "#version 310 es\nprecision highp float;\n"
        } else {
            "#version 430 core\n"
        }
    }

    pub fn bind_current(&self) -> Result<EglContextGuard<'_>> {
        let lock_guard = self
            ._lock
            .lock()
            .map_err(|_| ScalixError::ExecutionFailed("EGL context mutex poisoned".to_string()))?;
        unsafe {
            let res =
                (self.egl_make_current)(self.display, self.surface, self.surface, self.context);
            if res == EGL_FALSE {
                let err_code = self.egl_get_error.map_or(0, |f| f());
                return Err(ScalixError::ExecutionFailed(format!(
                    "eglMakeCurrent failed to bind EGL context to active thread: {}",
                    format_egl_error(err_code)
                )));
            }
        }
        Ok(EglContextGuard {
            ctx: self,
            _lock_guard: lock_guard,
        })
    }
}

impl Drop for EglContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.egl_make_current)(
                self.display,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                EGL_NO_CONTEXT,
            );
            if self.surface != EGL_NO_SURFACE {
                if let Some(f) = self.egl_destroy_surface {
                    let _ = f(self.display, self.surface);
                }
                self.surface = EGL_NO_SURFACE;
            }
            if let Some(f) = self.egl_destroy_context {
                let _ = f(self.display, self.context);
            }
            if let Some(f) = self.egl_terminate {
                let _ = f(self.display);
            }

            if !self.gbm_dev.is_null() {
                if let Some(destroy) = self.gbm_destroy_device {
                    destroy(self.gbm_dev);
                }
                self.gbm_dev = std::ptr::null_mut();
            }
            if self.gbm_fd >= 0 {
                libc::close(self.gbm_fd);
                self.gbm_fd = -1;
            }

            if !self.gbm_lib.is_null() {
                libc::dlclose(self.gbm_lib);
                self.gbm_lib = std::ptr::null_mut();
            }
            if !self.gl_lib.is_null() {
                libc::dlclose(self.gl_lib);
                self.gl_lib = std::ptr::null_mut();
            }
            if !self.egl_lib.is_null() {
                libc::dlclose(self.egl_lib);
                self.egl_lib = std::ptr::null_mut();
            }
        }
    }
}
