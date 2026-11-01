//! Headless EGL Context and OpenGL Function Loader

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
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
pub const EGL_OPENGL_ES3_BIT_KHR: EGLint = 0x0040;
pub const EGL_OPENGL_BIT: EGLint = 0x0008;
pub const EGL_RED_SIZE: EGLint = 0x3024;
pub const EGL_GREEN_SIZE: EGLint = 0x3023;
pub const EGL_BLUE_SIZE: EGLint = 0x3022;
pub const EGL_ALPHA_SIZE: EGLint = 0x3021;
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

pub const EGL_WIDTH: EGLint = 0x3057;
pub const EGL_HEIGHT: EGLint = 0x3056;
pub const EGL_EXTENSIONS: EGLint = 0x3055;

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
pub const GL_TEXTURE_MAX_LEVEL: u32 = 0x813D;

pub const GL_FRAMEBUFFER: u32 = 0x8D40;
pub const GL_READ_FRAMEBUFFER: u32 = 0x8CA8;
pub const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
pub const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;

pub const GL_FLOAT: u32 = 0x1406;
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
pub const GL_UNPACK_ROW_LENGTH: u32 = 0x0CF2;
pub const GL_PACK_ROW_LENGTH: u32 = 0x0D02;
pub const GL_PACK_ALIGNMENT: u32 = 0x0D05;
pub const GL_TEXTURE0: u32 = 0x84C0;
pub const GL_WRITE_ONLY: u32 = 0x88B9;
pub const GL_FRAMEBUFFER_BARRIER_BIT: u32 = 0x00000400;

#[allow(non_snake_case)]
pub struct GlFunctions {
    pub glGenTextures: unsafe extern "C" fn(i32, *mut u32),
    pub glBindTexture: unsafe extern "C" fn(u32, u32),
    pub glTexImage2D: unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    pub glTexStorage2D: Option<unsafe extern "C" fn(u32, i32, u32, i32, i32)>,
    pub glTexSubImage2D:
        unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    pub glTexParameteri: unsafe extern "C" fn(u32, u32, i32),
    pub glDeleteTextures: unsafe extern "C" fn(i32, *const u32),
    pub glActiveTexture: unsafe extern "C" fn(u32),
    pub glGenerateMipmap: unsafe extern "C" fn(u32),

    pub glGenFramebuffers: unsafe extern "C" fn(i32, *mut u32),
    pub glBindFramebuffer: unsafe extern "C" fn(u32, u32),
    pub glFramebufferTexture2D: unsafe extern "C" fn(u32, u32, u32, u32, i32),
    pub glDeleteFramebuffers: unsafe extern "C" fn(i32, *const u32),
    pub glBlitFramebuffer: unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32),
    pub glReadPixels: unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void),
    pub glViewport: unsafe extern "C" fn(i32, i32, i32, i32),
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
    pub glUniform2f: unsafe extern "C" fn(i32, f32, f32),

    pub glGenVertexArrays: unsafe extern "C" fn(i32, *mut u32),
    pub glBindVertexArray: unsafe extern "C" fn(u32),
    pub glGenBuffers: unsafe extern "C" fn(i32, *mut u32),
    pub glBindBuffer: unsafe extern "C" fn(u32, u32),
    pub glBufferData: unsafe extern "C" fn(u32, isize, *const c_void, u32),
    pub glEnableVertexAttribArray: unsafe extern "C" fn(u32),
    pub glVertexAttribPointer: unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void),
    pub glDrawArrays: unsafe extern "C" fn(u32, i32, i32),

    pub glFinish: unsafe extern "C" fn(),
    pub glGetString: unsafe extern "C" fn(u32) -> *const c_char,

    // Optional Compute
    pub glDispatchCompute: Option<unsafe extern "C" fn(u32, u32, u32)>,
    pub glBindImageTexture: Option<unsafe extern "C" fn(u32, u32, i32, u8, i32, u32, u32)>,
    pub glMemoryBarrier: Option<unsafe extern "C" fn(u32)>,

    // Hardware Synchronization Fences (GL 3.2+ / GLES 3.0+)

    // Buffer Mapping

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

/// An owned `dlopen` handle, closed on drop.
struct DlLib(NonNull<c_void>);

impl DlLib {
    /// Opens the first library in `names` that loads.
    unsafe fn open_first(names: &[&CStr], flags: c_int) -> Option<Self> {
        names
            .iter()
            .find_map(|name| NonNull::new(libc::dlopen(name.as_ptr(), flags)))
            .map(Self)
    }

    unsafe fn sym(&self, name: &CStr) -> *mut c_void {
        libc::dlsym(self.0.as_ptr(), name.as_ptr())
    }
}

impl Drop for DlLib {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.0.as_ptr());
        }
    }
}

/// A `gbm_device` and the render-node descriptor it was created on, released in that order.
struct GbmDevice {
    dev: NonNull<c_void>,
    destroy: Option<GbmDestroyDeviceFn>,
    _fd: OwnedFd,
}

impl Drop for GbmDevice {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy {
            unsafe { destroy(self.dev.as_ptr()) };
        }
    }
}

/// Reinterprets a symbol address as the function pointer type `F`; `None` when null.
///
/// # Safety
/// `F` must be the symbol's real signature.
unsafe fn fn_ptr<F: Copy>(ptr: *mut c_void) -> Option<F> {
    const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<*mut c_void>()) };
    (!ptr.is_null()).then(|| std::mem::transmute_copy::<*mut c_void, F>(&ptr))
}

fn unavailable() -> ScalixError {
    ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL)
}

pub struct EglContext {
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
    /// `glTexStorage2D` is usable (GL 4.2 / ES 3.0); see [`GlStagingSlot::ensure_dst_texture`].
    ///
    /// [`GlStagingSlot::ensure_dst_texture`]: super::ring::GlStagingSlot::ensure_dst_texture
    pub has_tex_storage: bool,
    pub gl_version: String,
    _lock: Mutex<()>,
    // Dropped after `Drop::drop` (which terminates a GBM display) in declaration order:
    // the GBM device, then the libraries its functions come from.
    gbm: Option<GbmDevice>,
    _gbm_lib: Option<DlLib>,
    _gl_lib: Option<DlLib>,
    _egl_lib: DlLib,
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
            let egl_lib = DlLib::open_first(
                &[c"libEGL.so.1", c"libEGL.so"],
                libc::RTLD_LAZY | libc::RTLD_GLOBAL,
            )
            .ok_or_else(unavailable)?;
            let gl_lib = DlLib::open_first(
                &[c"libOpenGL.so.0", c"libGL.so.1", c"libGLESv2.so.2"],
                libc::RTLD_LAZY | libc::RTLD_GLOBAL,
            );

            type EglGetProcAddressFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
            let egl_get_proc_address: Option<EglGetProcAddressFn> =
                fn_ptr(egl_lib.sym(c"eglGetProcAddress"));

            // Core EGL 1.0-1.4 symbols MUST be loaded via dlsym to avoid libglvnd dummy GL dispatch stubs
            let load_egl_core_symbol = |name: &CStr| -> *mut c_void { egl_lib.sym(name) };

            // EGL extension symbols (or EGL 1.5) can be resolved via dlsym or eglGetProcAddress
            let load_egl_ext_symbol = |name: &CStr| -> *mut c_void {
                let ptr = egl_lib.sym(name);
                match egl_get_proc_address {
                    Some(egpa) if ptr.is_null() => egpa(name.as_ptr()),
                    _ => ptr,
                }
            };

            // OpenGL client API symbols are resolved via eglGetProcAddress or libOpenGL/libGL/libGLESv2 dlsym
            let load_gl_symbol = |name: &CStr| -> *mut c_void {
                let mut ptr =
                    egl_get_proc_address.map_or(std::ptr::null_mut(), |egpa| egpa(name.as_ptr()));
                if ptr.is_null() {
                    if let Some(lib) = &gl_lib {
                        ptr = lib.sym(name);
                    }
                }
                if ptr.is_null() {
                    ptr = egl_lib.sym(name);
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

            let egl_get_display: Option<EglGetDisplayFn> =
                fn_ptr(load_egl_core_symbol(c"eglGetDisplay"));
            let egl_get_platform_display: Option<EglGetPlatformDisplayFn> =
                fn_ptr(load_egl_ext_symbol(c"eglGetPlatformDisplay"));
            let egl_get_platform_display_ext: Option<EglGetPlatformDisplayExtFn> =
                fn_ptr(load_egl_ext_symbol(c"eglGetPlatformDisplayEXT"));
            let egl_query_devices_ext: Option<EglQueryDevicesExtFn> =
                fn_ptr(load_egl_ext_symbol(c"eglQueryDevicesEXT"));

            type EglQueryDeviceStringExtFn =
                unsafe extern "C" fn(EGLDeviceEXT, EGLint) -> *const c_char;
            let egl_query_device_string_ext: Option<EglQueryDeviceStringExtFn> =
                fn_ptr(load_egl_ext_symbol(c"eglQueryDeviceStringEXT"));

            let egl_initialize: EglInitializeFn =
                fn_ptr(load_egl_core_symbol(c"eglInitialize")).ok_or_else(unavailable)?;

            let egl_bind_api: EglBindApiFn =
                fn_ptr(load_egl_core_symbol(c"eglBindAPI")).ok_or_else(unavailable)?;

            let egl_choose_config: EglChooseConfigFn =
                fn_ptr(load_egl_core_symbol(c"eglChooseConfig")).ok_or_else(unavailable)?;

            let egl_create_context: EglCreateContextFn =
                fn_ptr(load_egl_core_symbol(c"eglCreateContext")).ok_or_else(unavailable)?;

            let egl_make_current: EglMakeCurrentFn =
                fn_ptr(load_egl_core_symbol(c"eglMakeCurrent")).ok_or_else(unavailable)?;

            let egl_destroy_context: Option<EglDestroyContextFn> =
                fn_ptr(load_egl_core_symbol(c"eglDestroyContext"));
            let egl_create_pbuffer_surface: Option<EglCreatePbufferSurfaceFn> =
                fn_ptr(load_egl_core_symbol(c"eglCreatePbufferSurface"));
            let egl_destroy_surface: Option<EglDestroySurfaceFn> =
                fn_ptr(load_egl_core_symbol(c"eglDestroySurface"));
            let egl_get_error: Option<EglGetErrorFn> = fn_ptr(load_egl_core_symbol(c"eglGetError"));
            let egl_terminate: Option<EglTerminateFn> =
                fn_ptr(load_egl_core_symbol(c"eglTerminate"));

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

            type BoundDisplay = (EGLDisplay, EGLContext, EGLSurface, EGLint, EGLint);

            type GlGetStringFn = unsafe extern "C" fn(u32) -> *const c_char;
            let gl_get_string: Option<GlGetStringFn> = fn_ptr(load_gl_symbol(c"glGetString"));
            // Mesa silently falls back to a CPU rasterizer (e.g. surfaceless without DRM), so the
            // bound renderer itself is inspected rather than trusting the platform.
            let current_renderer_is_software = || -> bool {
                let Some(get_string) = gl_get_string else {
                    return false;
                };
                let p = get_string(0x1F01); // GL_RENDERER
                if p.is_null() {
                    return false;
                }
                let renderer = std::ffi::CStr::from_ptr(p).to_bytes();
                [
                    &b"llvmpipe"[..],
                    b"softpipe",
                    b"swrast",
                    b"Software Rasterizer",
                ]
                .iter()
                .any(|needle| renderer.windows(needle.len()).any(|w| w == *needle))
            };

            // NOTE: EGLDisplay handles are process-wide singletons per (platform, native display).
            // Device/surfaceless/default displays are shared with every other EglContext (and the
            // host application), so eglTerminate on them invalidates those contexts. Only displays
            // backed by our own private gbm_device are terminated. Action: none.
            let terminate_private = |display: EGLDisplay, is_private: bool| {
                if is_private {
                    if let Some(f) = egl_terminate {
                        let _ = f(display);
                    }
                }
            };

            // Initializes `cand_display`, creates a context and binds it (surfaceless, then 1x1
            // PBuffer). Software renderers are rejected unless `allow_software`; `is_private` marks a
            // display backed by our own gbm_device (the only kind safe to eglTerminate). On failure every
            // per-candidate resource is released and None returned.
            let try_display = |cand_display: EGLDisplay,
                               cand_desc: &str,
                               allow_software: bool,
                               is_private: bool|
             -> Option<BoundDisplay> {
                let mut major: EGLint = 0;
                let mut minor: EGLint = 0;
                if egl_initialize(cand_display, &mut major, &mut minor) == EGL_FALSE {
                    return None;
                }

                let _ = egl_bind_api(EGL_OPENGL_API);

                // EGL_RENDERABLE_TYPE defaults to EGL_OPENGL_ES_BIT; without requesting
                // EGL_OPENGL_BIT the chosen config may reject desktop GL context creation
                // (EGL_BAD_CONFIG/EGL_BAD_MATCH) and silently force the GLES fallback.
                let config_attribs = [
                    EGL_SURFACE_TYPE,
                    EGL_PBUFFER_BIT,
                    EGL_RENDERABLE_TYPE,
                    EGL_OPENGL_BIT,
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
                    terminate_private(cand_display, is_private);
                    return None;
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
                    // The desktop-GL config may not be ES3-renderable; re-pick one that is.
                    let gles_config_attribs = [
                        EGL_SURFACE_TYPE,
                        EGL_PBUFFER_BIT,
                        EGL_RENDERABLE_TYPE,
                        EGL_OPENGL_ES3_BIT_KHR,
                        EGL_NONE,
                    ];
                    let mut gles_config: EGLConfig = std::ptr::null_mut();
                    let mut num_gles_configs: EGLint = 0;
                    if egl_choose_config(
                        cand_display,
                        gles_config_attribs.as_ptr(),
                        &mut gles_config,
                        1,
                        &mut num_gles_configs,
                    ) == EGL_TRUE
                        && num_gles_configs > 0
                    {
                        config = gles_config;
                    }
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
                    terminate_private(cand_display, is_private);
                    return None;
                }

                // Try binding: 1. Surfaceless (EGL_NO_SURFACE), 2. 1x1 Offscreen PBuffer
                let mut pbuffer = EGL_NO_SURFACE;
                let mut bound =
                    egl_make_current(cand_display, EGL_NO_SURFACE, EGL_NO_SURFACE, cand_context)
                        == EGL_TRUE;
                if !bound {
                    let pbuffer_attribs = [EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE];
                    pbuffer = egl_create_pbuffer_surface
                        .map(|create| create(cand_display, config, pbuffer_attribs.as_ptr()))
                        .unwrap_or(EGL_NO_SURFACE);
                    bound = pbuffer != EGL_NO_SURFACE
                        && egl_make_current(cand_display, pbuffer, pbuffer, cand_context)
                            == EGL_TRUE;
                }

                if bound {
                    if allow_software || !current_renderer_is_software() {
                        log::debug!(
                            "Successfully bound headless EGL display via {} ({})",
                            cand_desc,
                            if pbuffer == EGL_NO_SURFACE {
                                "surfaceless"
                            } else {
                                "1x1 PBuffer"
                            }
                        );
                        return Some((cand_display, cand_context, pbuffer, major, minor));
                    }
                    log::debug!(
                        "Skipping software renderer on {} (hardware pass)",
                        cand_desc
                    );
                    let _ = egl_make_current(
                        cand_display,
                        EGL_NO_SURFACE,
                        EGL_NO_SURFACE,
                        EGL_NO_CONTEXT,
                    );
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
                terminate_private(cand_display, is_private);
                None
            };

            // Mesa always enumerates a CPU rasterizer (llvmpipe) device advertising
            // EGL_MESA_device_software, usually first. It is only a last resort.
            let is_software_device = |device: EGLDeviceEXT| -> bool {
                let Some(query) = egl_query_device_string_ext else {
                    return false;
                };
                let exts = query(device, EGL_EXTENSIONS);
                !exts.is_null()
                    && std::ffi::CStr::from_ptr(exts)
                        .to_bytes()
                        .split(|&b| b == b' ')
                        .any(|e| e == b"EGL_MESA_device_software")
            };

            let mut selected: Option<BoundDisplay> = None;
            let mut software_devices: Vec<EGLDeviceEXT> = Vec::new();

            // Strategy 1: hardware EGL devices (NVIDIA / AMD / Intel), software devices deferred
            if let Some(query_devices) = egl_query_devices_ext {
                let mut devices: [EGLDeviceEXT; 16] = [std::ptr::null_mut(); 16];
                let mut num_devices: EGLint = 0;
                if query_devices(16, devices.as_mut_ptr(), &mut num_devices) == EGL_TRUE {
                    let count = (num_devices.max(0) as usize).min(devices.len());
                    for &device in &devices[..count] {
                        if is_software_device(device) {
                            software_devices.push(device);
                            continue;
                        }
                        let d = get_platform_display(EGL_PLATFORM_DEVICE_EXT, device);
                        if d == EGL_NO_DISPLAY {
                            continue;
                        }
                        selected = try_display(d, "EGL_PLATFORM_DEVICE_EXT", false, false);
                        if selected.is_some() {
                            break;
                        }
                    }
                }
            }

            // Strategies 2/3: Surfaceless MESA (hardware render node when DRM exists), then the
            // default display (X11 / Wayland / Android; e.g. d3d12 on WSLg).
            let try_shared_displays = |allow_software: bool| -> Option<BoundDisplay> {
                let d = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY);
                if d != EGL_NO_DISPLAY {
                    if let Some(b) =
                        try_display(d, "EGL_PLATFORM_SURFACELESS_MESA", allow_software, false)
                    {
                        return Some(b);
                    }
                }
                let d = egl_get_display.map_or(EGL_NO_DISPLAY, |f| f(EGL_DEFAULT_DISPLAY));
                if d == EGL_NO_DISPLAY {
                    return None;
                }
                try_display(d, "eglGetDisplay default", allow_software, false)
            };
            if selected.is_none() {
                selected = try_shared_displays(false);
            }

            // Strategy 4: GBM on DRM render nodes, opened lazily. Primary nodes (/dev/dri/cardN)
            // are skipped: opening one on a headless host can implicitly grant DRM master.
            #[cfg(target_os = "linux")]
            let gbm_lib = if selected.is_none() {
                DlLib::open_first(
                    &[c"libgbm.so.1", c"libgbm.so"],
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            } else {
                None
            };
            #[cfg(not(target_os = "linux"))]
            let gbm_lib: Option<DlLib> = None;
            let gbm_destroy_device: Option<GbmDestroyDeviceFn> = gbm_lib
                .as_ref()
                .and_then(|lib| fn_ptr(lib.sym(c"gbm_device_destroy")));
            // Declared after `gbm_lib` so an early return drops the device before unloading libgbm.
            #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
            let mut selected_gbm: Option<GbmDevice> = None;

            #[cfg(target_os = "linux")]
            if let (None, Some(lib)) = (&selected, &gbm_lib) {
                type GbmCreateDeviceFn = unsafe extern "C" fn(i32) -> *mut c_void;
                if let Some(gbm_create_device) =
                    fn_ptr::<GbmCreateDeviceFn>(lib.sym(c"gbm_create_device"))
                {
                    for &path in crate::dma::linux_drm::DRM_CANDIDATE_PATHS
                        .iter()
                        .filter(|p| p.contains("renderD"))
                    {
                        let Ok(c_path) = CString::new(path) else {
                            continue;
                        };
                        let raw_fd = libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
                        if raw_fd < 0 {
                            continue;
                        }
                        let fd = OwnedFd::from_raw_fd(raw_fd);
                        let Some(dev) = NonNull::new(gbm_create_device(fd.as_raw_fd())) else {
                            continue;
                        };
                        let device = GbmDevice {
                            dev,
                            destroy: gbm_destroy_device,
                            _fd: fd,
                        };
                        let d = get_platform_display(EGL_PLATFORM_GBM_KHR, device.dev.as_ptr());
                        if d != EGL_NO_DISPLAY {
                            selected = try_display(d, "EGL_PLATFORM_GBM_KHR", false, true);
                        }
                        if selected.is_some() {
                            selected_gbm = Some(device);
                            break;
                        }
                    }
                }
            }

            // Strategy 5: software renderers as last resort (CI, VMs, LIBGL_ALWAYS_SOFTWARE)
            if selected.is_none() {
                selected = try_shared_displays(true);
            }
            if selected.is_none() {
                for &device in &software_devices {
                    let d = get_platform_display(EGL_PLATFORM_DEVICE_EXT, device);
                    if d == EGL_NO_DISPLAY {
                        continue;
                    }
                    selected = try_display(d, "EGL_PLATFORM_DEVICE_EXT (software)", true, false);
                    if selected.is_some() {
                        break;
                    }
                }
            }

            let (display, context, surface, bound_major, bound_minor) =
                selected.unwrap_or((EGL_NO_DISPLAY, EGL_NO_CONTEXT, EGL_NO_SURFACE, 0, 0));

            if display == EGL_NO_DISPLAY || context == EGL_NO_CONTEXT {
                return Err(unavailable());
            }

            log::debug!("Initialized headless EGL version {bound_major}.{bound_minor}");

            // Releases everything acquired so far if GL function loading below bails out early.
            let rollback = InitRollback {
                display,
                context,
                surface,
                egl_make_current,
                egl_destroy_context,
                egl_destroy_surface,
                egl_terminate,
                terminate_display: selected_gbm.is_some(),
            };

            macro_rules! get_gl_fn {
                ($name:expr, $t:ty) => {{
                    fn_ptr::<$t>(load_gl_symbol($name)).ok_or_else(|| {
                        ScalixError::ExecutionFailed(format!("Missing GL symbol: {:?}", $name))
                    })?
                }};
            }

            macro_rules! get_optional_gl_fn {
                ($name:expr, $t:ty) => {{
                    fn_ptr::<$t>(load_gl_symbol($name))
                }};
            }

            let gl = GlFunctions {
                glGenTextures: get_gl_fn!(c"glGenTextures", unsafe extern "C" fn(i32, *mut u32)),
                glBindTexture: get_gl_fn!(c"glBindTexture", unsafe extern "C" fn(u32, u32)),
                glTexImage2D: get_gl_fn!(
                    c"glTexImage2D",
                    unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
                ),
                glTexStorage2D: get_optional_gl_fn!(
                    c"glTexStorage2D",
                    unsafe extern "C" fn(u32, i32, u32, i32, i32)
                ),
                glTexSubImage2D: get_gl_fn!(
                    c"glTexSubImage2D",
                    unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
                ),
                glTexParameteri: get_gl_fn!(
                    c"glTexParameteri",
                    unsafe extern "C" fn(u32, u32, i32)
                ),
                glDeleteTextures: get_gl_fn!(
                    c"glDeleteTextures",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glActiveTexture: get_gl_fn!(c"glActiveTexture", unsafe extern "C" fn(u32)),
                glGenerateMipmap: get_gl_fn!(c"glGenerateMipmap", unsafe extern "C" fn(u32)),

                glGenFramebuffers: get_gl_fn!(
                    c"glGenFramebuffers",
                    unsafe extern "C" fn(i32, *mut u32)
                ),
                glBindFramebuffer: get_gl_fn!(c"glBindFramebuffer", unsafe extern "C" fn(u32, u32)),
                glFramebufferTexture2D: get_gl_fn!(
                    c"glFramebufferTexture2D",
                    unsafe extern "C" fn(u32, u32, u32, u32, i32)
                ),
                glDeleteFramebuffers: get_gl_fn!(
                    c"glDeleteFramebuffers",
                    unsafe extern "C" fn(i32, *const u32)
                ),
                glBlitFramebuffer: get_gl_fn!(
                    c"glBlitFramebuffer",
                    unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32)
                ),
                glReadPixels: get_gl_fn!(
                    c"glReadPixels",
                    unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void)
                ),
                glViewport: get_gl_fn!(c"glViewport", unsafe extern "C" fn(i32, i32, i32, i32)),
                glPixelStorei: get_gl_fn!(c"glPixelStorei", unsafe extern "C" fn(u32, i32)),

                glCreateShader: get_gl_fn!(c"glCreateShader", unsafe extern "C" fn(u32) -> u32),
                glShaderSource: get_gl_fn!(
                    c"glShaderSource",
                    unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32)
                ),
                glCompileShader: get_gl_fn!(c"glCompileShader", unsafe extern "C" fn(u32)),
                glGetShaderiv: get_gl_fn!(
                    c"glGetShaderiv",
                    unsafe extern "C" fn(u32, u32, *mut i32)
                ),
                glGetShaderInfoLog: get_gl_fn!(
                    c"glGetShaderInfoLog",
                    unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)
                ),
                glCreateProgram: get_gl_fn!(c"glCreateProgram", unsafe extern "C" fn() -> u32),
                glAttachShader: get_gl_fn!(c"glAttachShader", unsafe extern "C" fn(u32, u32)),
                glLinkProgram: get_gl_fn!(c"glLinkProgram", unsafe extern "C" fn(u32)),
                glGetProgramiv: get_gl_fn!(
                    c"glGetProgramiv",
                    unsafe extern "C" fn(u32, u32, *mut i32)
                ),
                glGetProgramInfoLog: get_gl_fn!(
                    c"glGetProgramInfoLog",
                    unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)
                ),
                glUseProgram: get_gl_fn!(c"glUseProgram", unsafe extern "C" fn(u32)),
                glDeleteProgram: get_gl_fn!(c"glDeleteProgram", unsafe extern "C" fn(u32)),
                glDeleteShader: get_gl_fn!(c"glDeleteShader", unsafe extern "C" fn(u32)),
                glGetUniformLocation: get_gl_fn!(
                    c"glGetUniformLocation",
                    unsafe extern "C" fn(u32, *const c_char) -> i32
                ),
                glUniform1i: get_gl_fn!(c"glUniform1i", unsafe extern "C" fn(i32, i32)),
                glUniform2f: get_gl_fn!(c"glUniform2f", unsafe extern "C" fn(i32, f32, f32)),

                glGenVertexArrays: get_gl_fn!(
                    c"glGenVertexArrays",
                    unsafe extern "C" fn(i32, *mut u32)
                ),
                glBindVertexArray: get_gl_fn!(c"glBindVertexArray", unsafe extern "C" fn(u32)),
                glGenBuffers: get_gl_fn!(c"glGenBuffers", unsafe extern "C" fn(i32, *mut u32)),
                glBindBuffer: get_gl_fn!(c"glBindBuffer", unsafe extern "C" fn(u32, u32)),
                glBufferData: get_gl_fn!(
                    c"glBufferData",
                    unsafe extern "C" fn(u32, isize, *const c_void, u32)
                ),
                glEnableVertexAttribArray: get_gl_fn!(
                    c"glEnableVertexAttribArray",
                    unsafe extern "C" fn(u32)
                ),
                glVertexAttribPointer: get_gl_fn!(
                    c"glVertexAttribPointer",
                    unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void)
                ),
                glDrawArrays: get_gl_fn!(c"glDrawArrays", unsafe extern "C" fn(u32, i32, i32)),

                glFinish: get_gl_fn!(c"glFinish", unsafe extern "C" fn()),
                glGetString: get_gl_fn!(c"glGetString", unsafe extern "C" fn(u32) -> *const c_char),

                glDispatchCompute: get_optional_gl_fn!(
                    c"glDispatchCompute",
                    unsafe extern "C" fn(u32, u32, u32)
                ),
                glBindImageTexture: get_optional_gl_fn!(
                    c"glBindImageTexture",
                    unsafe extern "C" fn(u32, u32, i32, u8, i32, u32, u32)
                ),
                glMemoryBarrier: get_optional_gl_fn!(c"glMemoryBarrier", unsafe extern "C" fn(u32)),
                glGetError: get_optional_gl_fn!(c"glGetError", unsafe extern "C" fn() -> u32),
            };

            std::mem::forget(rollback);

            let version_ptr = (gl.glGetString)(0x1F02); // GL_VERSION = 0x1F02
            let version_str = if !version_ptr.is_null() {
                std::ffi::CStr::from_ptr(version_ptr)
                    .to_string_lossy()
                    .to_string()
            } else {
                String::new()
            };
            let is_gles = version_str.starts_with("OpenGL ES");
            // Symbol presence proves nothing: libglvnd's eglGetProcAddress returns dispatch stubs
            // for every gl* name, so compute is gated on the context version (GL 4.3 / ES 3.1).
            let min_compute_version = if is_gles { (3, 1) } else { (4, 3) };
            let gl_version = parse_gl_version(&version_str);
            let is_compute_supported = gl.glDispatchCompute.is_some()
                && gl.glBindImageTexture.is_some()
                && gl_version >= min_compute_version;
            let min_tex_storage_version = if is_gles { (3, 0) } else { (4, 2) };
            let has_tex_storage =
                gl.glTexStorage2D.is_some() && gl_version >= min_tex_storage_version;

            log::debug!(
                "EGL/GL Context Initialized: Version '{}', is_gles = {}, compute = {}, offscreen_pbuffer = {}",
                version_str,
                is_gles,
                is_compute_supported,
                surface != EGL_NO_SURFACE
            );

            // Unbind from creation thread so worker threads can bind it
            let _ = egl_make_current(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

            // libgbm is only needed while a GBM device is kept.
            let gbm_lib = gbm_lib.filter(|_| selected_gbm.is_some());

            Ok(Self {
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
                has_tex_storage,
                gl_version: version_str,
                _lock: Mutex::new(()),
                gbm: selected_gbm,
                _gbm_lib: gbm_lib,
                _gl_lib: gl_lib,
                _egl_lib: egl_lib,
            })
        }
    }

    pub fn shader_header(&self) -> &'static str {
        if self.is_gles {
            "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n"
        } else {
            "#version 330 core\n"
        }
    }

    pub fn compute_shader_header(&self) -> &'static str {
        if self.is_gles {
            "#version 310 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp image2D;\n"
        } else {
            "#version 430 core\n"
        }
    }

    /// Drains the GL error queue, failing with the first recorded error.
    ///
    /// Must be called with this context current (i.e. while holding an [`EglContextGuard`]).
    pub fn check_gl_error(&self, stage: &str) -> Result<()> {
        // GL keeps one flag per error kind; a bounded drain avoids spinning on a lost context.
        const MAX_QUEUED_GL_ERRORS: usize = 8;
        let Some(get_error) = self.gl.glGetError else {
            return Ok(());
        };
        let mut first = 0;
        for _ in 0..MAX_QUEUED_GL_ERRORS {
            let code = unsafe { get_error() };
            if code == 0 {
                break;
            }
            if first == 0 {
                first = code;
            }
        }
        if first == 0 {
            return Ok(());
        }
        Err(ScalixError::ExecutionFailed(format!(
            "OpenGL {stage} failed: {}",
            format_gl_error(first)
        )))
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

/// Parses `(major, minor)` from a `GL_VERSION` string such as `"4.5 (Core Profile) Mesa 23.2"`
/// or `"OpenGL ES 3.1 Mesa 23.2"`. Returns `(0, 0)` when unparseable.
fn parse_gl_version(version: &str) -> (u32, u32) {
    let numeric = version.trim_start_matches("OpenGL ES").trim_start();
    let mut parts = numeric
        .split(|c: char| !c.is_ascii_digit())
        .map(|p| p.parse::<u32>().ok());
    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(major), Some(minor)) => (major, minor),
        _ => (0, 0),
    }
}

/// Scope guard undoing a partially constructed [`EglContext`]; defused with `mem::forget`.
///
/// Declared after the libraries and the GBM device, so on an early return it runs first and the
/// display is terminated before those are released.
struct InitRollback {
    display: EGLDisplay,
    context: EGLContext,
    surface: EGLSurface,
    egl_make_current: EglMakeCurrentFn,
    egl_destroy_context: Option<EglDestroyContextFn>,
    egl_destroy_surface: Option<EglDestroySurfaceFn>,
    egl_terminate: Option<EglTerminateFn>,
    /// The display is backed by our private GBM device (the only kind safe to terminate).
    terminate_display: bool,
}

impl Drop for InitRollback {
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
            }
            if let Some(f) = self.egl_destroy_context {
                let _ = f(self.display, self.context);
            }
            if self.terminate_display {
                if let Some(f) = self.egl_terminate {
                    let _ = f(self.display);
                }
            }
        }
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
            // Shared displays must stay initialized for sibling contexts; see EglContext::new.
            // The GBM device and the libraries are released afterwards by their field drops.
            if self.gbm.is_some() {
                if let Some(f) = self.egl_terminate {
                    let _ = f(self.display);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_gl_version;

    #[test]
    fn parses_desktop_and_es_versions() {
        assert_eq!(parse_gl_version("4.5 (Core Profile) Mesa 23.2.1"), (4, 5));
        assert_eq!(parse_gl_version("4.6.0 NVIDIA 535.183.01"), (4, 6));
        assert_eq!(parse_gl_version("3.3 (Core Profile) Mesa 22.0.1"), (3, 3));
        assert_eq!(parse_gl_version("OpenGL ES 3.1 Mesa 23.2.1"), (3, 1));
        assert_eq!(parse_gl_version("OpenGL ES 3.0 V@0502.0"), (3, 0));
        assert_eq!(parse_gl_version(""), (0, 0));
    }

    #[test]
    fn compute_threshold_ordering() {
        assert!(parse_gl_version("3.3 (Core Profile) Mesa 22.0.1") < (4, 3));
        assert!(parse_gl_version("4.3.0 NVIDIA 470.256.02") >= (4, 3));
        assert!(parse_gl_version("OpenGL ES 3.0 Mesa") < (3, 1));
    }
}
