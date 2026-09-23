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
    pub glTexSubImage2D: unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
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
}

pub struct EglContext {
    egl_lib: *mut c_void,
    gl_lib: *mut c_void,
    display: EGLDisplay,
    context: EGLContext,
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
            type EglMakeCurrentFn = unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;
            if let Some(ptr) = self.ctx.get_proc_address("eglMakeCurrent") {
                let egl_make_current: EglMakeCurrentFn = std::mem::transmute(ptr);
                let _ = egl_make_current(self.ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            }
        }
    }
}

unsafe impl Send for EglContext {}
unsafe impl Sync for EglContext {}

impl EglContext {
    pub fn new() -> Result<Self> {
        unsafe {
            let egl_lib = libc::dlopen(c"libEGL.so.1".as_ptr() as *const c_char, libc::RTLD_LAZY | libc::RTLD_GLOBAL);
            let egl_lib = if egl_lib.is_null() {
                libc::dlopen(c"libEGL.so".as_ptr() as *const c_char, libc::RTLD_LAZY | libc::RTLD_GLOBAL)
            } else {
                egl_lib
            };

            if egl_lib.is_null() {
                return Err(ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL));
            }

            let gl_lib = libc::dlopen(c"libGL.so.1".as_ptr() as *const c_char, libc::RTLD_LAZY | libc::RTLD_GLOBAL);
            let gl_lib = if gl_lib.is_null() {
                libc::dlopen(c"libGLESv2.so.2".as_ptr() as *const c_char, libc::RTLD_LAZY | libc::RTLD_GLOBAL)
            } else {
                gl_lib
            };

            type EglGetProcAddressFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
            let egl_get_proc_address_ptr = libc::dlsym(egl_lib, c"eglGetProcAddress".as_ptr() as *const c_char);
            if egl_get_proc_address_ptr.is_null() {
                libc::dlclose(egl_lib);
                if !gl_lib.is_null() { libc::dlclose(gl_lib); }
                return Err(ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL));
            }
            let egl_get_proc_address: EglGetProcAddressFn = std::mem::transmute(egl_get_proc_address_ptr);

            let load_symbol = |name: &str| -> *mut c_void {
                let c_name = CString::new(name).unwrap();
                let mut ptr = egl_get_proc_address(c_name.as_ptr());
                if ptr.is_null() && !gl_lib.is_null() {
                    ptr = libc::dlsym(gl_lib, c_name.as_ptr());
                }
                if ptr.is_null() {
                    ptr = libc::dlsym(egl_lib, c_name.as_ptr());
                }
                ptr
            };

            type EglGetDisplayFn = unsafe extern "C" fn(EGLDisplay) -> EGLDisplay;
            type EglInitializeFn = unsafe extern "C" fn(EGLDisplay, *mut EGLint, *mut EGLint) -> EGLBoolean;
            type EglBindApiFn = unsafe extern "C" fn(u32) -> EGLBoolean;
            type EglChooseConfigFn = unsafe extern "C" fn(EGLDisplay, *const EGLint, *mut EGLConfig, EGLint, *mut EGLint) -> EGLBoolean;
            type EglCreateContextFn = unsafe extern "C" fn(EGLDisplay, EGLConfig, EGLContext, *const EGLint) -> EGLContext;
            type EglMakeCurrentFn = unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;

            let egl_get_display: EglGetDisplayFn = std::mem::transmute(load_symbol("eglGetDisplay"));
            let egl_initialize: EglInitializeFn = std::mem::transmute(load_symbol("eglInitialize"));
            let egl_bind_api: EglBindApiFn = std::mem::transmute(load_symbol("eglBindAPI"));
            let egl_choose_config: EglChooseConfigFn = std::mem::transmute(load_symbol("eglChooseConfig"));
            let egl_create_context: EglCreateContextFn = std::mem::transmute(load_symbol("eglCreateContext"));
            let egl_make_current: EglMakeCurrentFn = std::mem::transmute(load_symbol("eglMakeCurrent"));

            let display = egl_get_display(EGL_DEFAULT_DISPLAY);
            if display == EGL_NO_DISPLAY {
                libc::dlclose(egl_lib);
                if !gl_lib.is_null() { libc::dlclose(gl_lib); }
                return Err(ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL));
            }

            let mut major: EGLint = 0;
            let mut minor: EGLint = 0;
            if egl_initialize(display, &mut major, &mut minor) == EGL_FALSE {
                libc::dlclose(egl_lib);
                if !gl_lib.is_null() { libc::dlclose(gl_lib); }
                return Err(ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL));
            }

            log::debug!("Initialized headless EGL version {major}.{minor}");

            let _ = egl_bind_api(EGL_OPENGL_API);

            let config_attribs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_NONE,
            ];

            let mut config: EGLConfig = std::ptr::null_mut();
            let mut num_configs: EGLint = 0;
            if egl_choose_config(display, config_attribs.as_ptr(), &mut config, 1, &mut num_configs) == EGL_FALSE || num_configs < 1 {
                // Fallback to minimal config
                let minimal_attribs = [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE];
                let _ = egl_choose_config(display, minimal_attribs.as_ptr(), &mut config, 1, &mut num_configs);
            }

            let context_attribs_gl43 = [
                EGL_CONTEXT_MAJOR_VERSION, 4,
                EGL_CONTEXT_MINOR_VERSION, 3,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            ];
            let context_attribs_gl33 = [
                EGL_CONTEXT_MAJOR_VERSION, 3,
                EGL_CONTEXT_MINOR_VERSION, 3,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            ];

            let mut context = egl_create_context(display, config, EGL_NO_CONTEXT, context_attribs_gl43.as_ptr());
            if context == EGL_NO_CONTEXT {
                context = egl_create_context(display, config, EGL_NO_CONTEXT, context_attribs_gl33.as_ptr());
            }
            if context == EGL_NO_CONTEXT {
                // Fallback to GLES 3.1 / 3.0 / default
                let _ = egl_bind_api(EGL_OPENGL_ES_API);
                let gles_attribs_31 = [
                    EGL_CONTEXT_CLIENT_VERSION, 3,
                    EGL_CONTEXT_MINOR_VERSION, 1,
                    EGL_NONE,
                ];
                context = egl_create_context(display, config, EGL_NO_CONTEXT, gles_attribs_31.as_ptr());
                if context == EGL_NO_CONTEXT {
                    let gles_attribs_30 = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
                    context = egl_create_context(display, config, EGL_NO_CONTEXT, gles_attribs_30.as_ptr());
                }
                if context == EGL_NO_CONTEXT {
                    context = egl_create_context(display, config, EGL_NO_CONTEXT, std::ptr::null());
                }
            }

            if context == EGL_NO_CONTEXT {
                libc::dlclose(egl_lib);
                if !gl_lib.is_null() { libc::dlclose(gl_lib); }
                return Err(ScalixError::BackendUnavailable(crate::types::BackendType::OpenGL));
            }

            if egl_make_current(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context) == EGL_FALSE {
                log::warn!("eglMakeCurrent surfaceless returned false");
            }

            macro_rules! get_gl_fn {
                ($name:expr, $t:ty) => {{
                    let ptr = load_symbol($name);
                    if ptr.is_null() {
                        return Err(ScalixError::ExecutionFailed(format!("Missing GL symbol: {}", $name)));
                    }
                    std::mem::transmute::<*mut c_void, $t>(ptr)
                }};
            }

            macro_rules! get_optional_gl_fn {
                ($name:expr, $t:ty) => {{
                    let ptr = load_symbol($name);
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
                glTexImage2D: get_gl_fn!("glTexImage2D", unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)),
                glTexSubImage2D: get_gl_fn!("glTexSubImage2D", unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)),
                glTexParameteri: get_gl_fn!("glTexParameteri", unsafe extern "C" fn(u32, u32, i32)),
                glDeleteTextures: get_gl_fn!("glDeleteTextures", unsafe extern "C" fn(i32, *const u32)),
                glActiveTexture: get_gl_fn!("glActiveTexture", unsafe extern "C" fn(u32)),
                glGenerateMipmap: get_gl_fn!("glGenerateMipmap", unsafe extern "C" fn(u32)),

                glGenFramebuffers: get_gl_fn!("glGenFramebuffers", unsafe extern "C" fn(i32, *mut u32)),
                glBindFramebuffer: get_gl_fn!("glBindFramebuffer", unsafe extern "C" fn(u32, u32)),
                glFramebufferTexture2D: get_gl_fn!("glFramebufferTexture2D", unsafe extern "C" fn(u32, u32, u32, u32, i32)),
                glCheckFramebufferStatus: get_gl_fn!("glCheckFramebufferStatus", unsafe extern "C" fn(u32) -> u32),
                glDeleteFramebuffers: get_gl_fn!("glDeleteFramebuffers", unsafe extern "C" fn(i32, *const u32)),
                glBlitFramebuffer: get_gl_fn!("glBlitFramebuffer", unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32)),
                glReadPixels: get_gl_fn!("glReadPixels", unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void)),
                glViewport: get_gl_fn!("glViewport", unsafe extern "C" fn(i32, i32, i32, i32)),
                glClear: get_gl_fn!("glClear", unsafe extern "C" fn(u32)),
                glClearColor: get_gl_fn!("glClearColor", unsafe extern "C" fn(f32, f32, f32, f32)),
                glPixelStorei: get_gl_fn!("glPixelStorei", unsafe extern "C" fn(u32, i32)),

                glCreateShader: get_gl_fn!("glCreateShader", unsafe extern "C" fn(u32) -> u32),
                glShaderSource: get_gl_fn!("glShaderSource", unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32)),
                glCompileShader: get_gl_fn!("glCompileShader", unsafe extern "C" fn(u32)),
                glGetShaderiv: get_gl_fn!("glGetShaderiv", unsafe extern "C" fn(u32, u32, *mut i32)),
                glGetShaderInfoLog: get_gl_fn!("glGetShaderInfoLog", unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)),
                glCreateProgram: get_gl_fn!("glCreateProgram", unsafe extern "C" fn() -> u32),
                glAttachShader: get_gl_fn!("glAttachShader", unsafe extern "C" fn(u32, u32)),
                glLinkProgram: get_gl_fn!("glLinkProgram", unsafe extern "C" fn(u32)),
                glGetProgramiv: get_gl_fn!("glGetProgramiv", unsafe extern "C" fn(u32, u32, *mut i32)),
                glGetProgramInfoLog: get_gl_fn!("glGetProgramInfoLog", unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char)),
                glUseProgram: get_gl_fn!("glUseProgram", unsafe extern "C" fn(u32)),
                glDeleteProgram: get_gl_fn!("glDeleteProgram", unsafe extern "C" fn(u32)),
                glDeleteShader: get_gl_fn!("glDeleteShader", unsafe extern "C" fn(u32)),
                glGetUniformLocation: get_gl_fn!("glGetUniformLocation", unsafe extern "C" fn(u32, *const c_char) -> i32),
                glUniform1i: get_gl_fn!("glUniform1i", unsafe extern "C" fn(i32, i32)),
                glUniform1f: get_gl_fn!("glUniform1f", unsafe extern "C" fn(i32, f32)),
                glUniform2f: get_gl_fn!("glUniform2f", unsafe extern "C" fn(i32, f32, f32)),

                glGenVertexArrays: get_gl_fn!("glGenVertexArrays", unsafe extern "C" fn(i32, *mut u32)),
                glBindVertexArray: get_gl_fn!("glBindVertexArray", unsafe extern "C" fn(u32)),
                glDeleteVertexArrays: get_gl_fn!("glDeleteVertexArrays", unsafe extern "C" fn(i32, *const u32)),
                glGenBuffers: get_gl_fn!("glGenBuffers", unsafe extern "C" fn(i32, *mut u32)),
                glBindBuffer: get_gl_fn!("glBindBuffer", unsafe extern "C" fn(u32, u32)),
                glBufferData: get_gl_fn!("glBufferData", unsafe extern "C" fn(u32, isize, *const c_void, u32)),
                glDeleteBuffers: get_gl_fn!("glDeleteBuffers", unsafe extern "C" fn(i32, *const u32)),
                glEnableVertexAttribArray: get_gl_fn!("glEnableVertexAttribArray", unsafe extern "C" fn(u32)),
                glVertexAttribPointer: get_gl_fn!("glVertexAttribPointer", unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void)),
                glDrawArrays: get_gl_fn!("glDrawArrays", unsafe extern "C" fn(u32, i32, i32)),

                glFinish: get_gl_fn!("glFinish", unsafe extern "C" fn()),
                glFlush: get_gl_fn!("glFlush", unsafe extern "C" fn()),
                glGetString: get_gl_fn!("glGetString", unsafe extern "C" fn(u32) -> *const c_char),

                glDispatchCompute: get_optional_gl_fn!("glDispatchCompute", unsafe extern "C" fn(u32, u32, u32)),
                glBindImageTexture: get_optional_gl_fn!("glBindImageTexture", unsafe extern "C" fn(u32, u32, i32, u8, i32, u32, u32)),
                glMemoryBarrier: get_optional_gl_fn!("glMemoryBarrier", unsafe extern "C" fn(u32)),

                glFenceSync: get_optional_gl_fn!("glFenceSync", unsafe extern "C" fn(u32, u32) -> GLsync),
                glClientWaitSync: get_optional_gl_fn!("glClientWaitSync", unsafe extern "C" fn(GLsync, u32, u64) -> u32),
                glWaitSync: get_optional_gl_fn!("glWaitSync", unsafe extern "C" fn(GLsync, u32, u64)),
                glDeleteSync: get_optional_gl_fn!("glDeleteSync", unsafe extern "C" fn(GLsync)),

                glMapBufferRange: get_optional_gl_fn!("glMapBufferRange", unsafe extern "C" fn(u32, isize, isize, u32) -> *mut c_void),
                glUnmapBuffer: get_optional_gl_fn!("glUnmapBuffer", unsafe extern "C" fn(u32) -> u8),
            };

            let is_compute_supported = gl.glDispatchCompute.is_some() && gl.glBindImageTexture.is_some();

            let version_ptr = (gl.glGetString)(0x1F02); // GL_VERSION = 0x1F02
            let version_str = if !version_ptr.is_null() {
                std::ffi::CStr::from_ptr(version_ptr).to_string_lossy().to_string()
            } else {
                String::new()
            };
            let is_gles = version_str.contains("OpenGL ES") || version_str.contains("ES");

            log::debug!(
                "EGL/GL Context Initialized: Version '{}', is_gles = {}, compute = {}",
                version_str,
                is_gles,
                is_compute_supported
            );

            // Unbind from creation thread so worker threads can bind it
            let _ = egl_make_current(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

            Ok(Self {
                egl_lib,
                gl_lib,
                display,
                context,
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
        let lock_guard = self._lock.lock().map_err(|_| {
            ScalixError::ExecutionFailed("EGL context mutex poisoned".to_string())
        })?;
        unsafe {
            type EglMakeCurrentFn = unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;
            if let Some(ptr) = self.get_proc_address("eglMakeCurrent") {
                let egl_make_current: EglMakeCurrentFn = std::mem::transmute(ptr);
                let _ = egl_make_current(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, self.context);
            }
        }
        Ok(EglContextGuard {
            ctx: self,
            _lock_guard: lock_guard,
        })
    }

    fn get_proc_address(&self, name: &str) -> Option<*mut c_void> {
        unsafe {
            type EglGetProcAddressFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
            let get_proc = libc::dlsym(self.egl_lib, c"eglGetProcAddress".as_ptr() as *const c_char);
            if !get_proc.is_null() {
                let f: EglGetProcAddressFn = std::mem::transmute(get_proc);
                let c_name = CString::new(name).unwrap();
                let ptr = f(c_name.as_ptr());
                if !ptr.is_null() { return Some(ptr); }
            }
            if !self.gl_lib.is_null() {
                let c_name = CString::new(name).unwrap();
                let ptr = libc::dlsym(self.gl_lib, c_name.as_ptr());
                if !ptr.is_null() { return Some(ptr); }
            }
            let c_name = CString::new(name).unwrap();
            let ptr = libc::dlsym(self.egl_lib, c_name.as_ptr());
            if !ptr.is_null() { Some(ptr) } else { None }
        }
    }
}

impl Drop for EglContext {
    fn drop(&mut self) {
        unsafe {
            type EglDestroyContextFn = unsafe extern "C" fn(EGLDisplay, EGLContext) -> EGLBoolean;
            type EglTerminateFn = unsafe extern "C" fn(EGLDisplay) -> EGLBoolean;
            type EglMakeCurrentFn = unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;

            if let Some(ptr) = self.get_proc_address("eglMakeCurrent") {
                let egl_make_current: EglMakeCurrentFn = std::mem::transmute(ptr);
                egl_make_current(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            }
            if let Some(ptr) = self.get_proc_address("eglDestroyContext") {
                let egl_destroy_context: EglDestroyContextFn = std::mem::transmute(ptr);
                egl_destroy_context(self.display, self.context);
            }
            if let Some(ptr) = self.get_proc_address("eglTerminate") {
                let egl_terminate: EglTerminateFn = std::mem::transmute(ptr);
                egl_terminate(self.display);
            }

            if !self.gl_lib.is_null() {
                libc::dlclose(self.gl_lib);
            }
            if !self.egl_lib.is_null() {
                libc::dlclose(self.egl_lib);
            }
        }
    }
}
