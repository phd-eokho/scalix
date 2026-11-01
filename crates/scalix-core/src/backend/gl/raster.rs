//! OpenGL Raster Shader Pipeline (Fullscreen Quad & Shaders)

use std::ffi::c_char;
use std::sync::Arc;
use std::time::Instant;

use super::blit::get_gl_format_tuple;
use super::context::*;
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

const VERTEX_SHADER_BODY: &str = include_str!("shaders/quad.vert");
const FRAGMENT_SHADER_BILINEAR_BODY: &str = include_str!("shaders/raster_bilinear.frag");
const FRAGMENT_SHADER_BICUBIC_BODY: &str = include_str!("shaders/raster_bicubic.frag");
const FRAGMENT_SHADER_AREA_BODY: &str = include_str!("shaders/raster_area.frag");
const FRAGMENT_SHADER_LANCZOS3_BODY: &str = include_str!("shaders/raster_lanczos3.frag");
const RESAMPLE_KERNELS: &str = include_str!("shaders/resample_kernels.glsl");

use super::ring::GlStagingRing;
use std::sync::Mutex;

/// A linked raster program with its per-frame uniform locations resolved once at build time.
#[derive(Clone, Copy)]
struct RasterProgram {
    id: u32,
    u_tex_size: i32,
    u_dst_size: i32,
}

pub struct GlRasterResizer {
    ctx: Arc<EglContext>,
    ring: Arc<Mutex<GlStagingRing>>,
    profiler: Arc<dyn Profiler>,
    prog_bilinear: Mutex<Option<RasterProgram>>,
    prog_bicubic: Mutex<Option<RasterProgram>>,
    prog_area: Mutex<Option<RasterProgram>>,
    prog_lanczos3: Mutex<Option<RasterProgram>>,
    mesh: Mutex<Option<(u32, u32)>>, // (vao, vbo)
}

impl GlRasterResizer {
    pub fn new(
        ctx: Arc<EglContext>,
        ring: Arc<Mutex<GlStagingRing>>,
        profiler: Arc<dyn Profiler>,
    ) -> Self {
        Self {
            ctx,
            ring,
            profiler,
            prog_bilinear: Mutex::new(None),
            prog_bicubic: Mutex::new(None),
            prog_area: Mutex::new(None),
            prog_lanczos3: Mutex::new(None),
            mesh: Mutex::new(None),
        }
    }

    fn ensure_mesh(&self) -> Result<(u32, u32)> {
        let mut mesh_guard = self
            .mesh
            .lock()
            .map_err(|_| ScalixError::ExecutionFailed("Failed to lock raster mesh".to_string()))?;
        if let Some(m) = *mesh_guard {
            return Ok(m);
        }

        let quad_vertices: [f32; 16] = [
            -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let gl = &self.ctx.gl;
        let mut vao = 0u32;
        let mut vbo = 0u32;
        unsafe {
            (gl.glGenVertexArrays)(1, &mut vao);
            (gl.glGenBuffers)(1, &mut vbo);

            (gl.glBindVertexArray)(vao);
            (gl.glBindBuffer)(GL_ARRAY_BUFFER, vbo);
            (gl.glBufferData)(
                GL_ARRAY_BUFFER,
                (quad_vertices.len() * std::mem::size_of::<f32>()) as isize,
                quad_vertices.as_ptr() as *const std::ffi::c_void,
                GL_STATIC_DRAW,
            );

            (gl.glEnableVertexAttribArray)(0);
            (gl.glVertexAttribPointer)(
                0,
                2,
                GL_FLOAT,
                0,
                (4 * std::mem::size_of::<f32>()) as i32,
                std::ptr::null(),
            );

            (gl.glEnableVertexAttribArray)(1);
            (gl.glVertexAttribPointer)(
                1,
                2,
                GL_FLOAT,
                0,
                (4 * std::mem::size_of::<f32>()) as i32,
                (2 * std::mem::size_of::<f32>()) as *const std::ffi::c_void,
            );
            (gl.glBindVertexArray)(0);
            (gl.glBindBuffer)(GL_ARRAY_BUFFER, 0);
        }

        let m = (vao, vbo);
        *mesh_guard = Some(m);
        Ok(m)
    }

    fn compile_shader(&self, shader_type: u32, source: &str) -> Result<u32> {
        let gl = &self.ctx.gl;
        unsafe {
            let shader = (gl.glCreateShader)(shader_type);
            // Explicit length: the source needs no NUL terminator or CString copy.
            let len = i32::try_from(source.len()).map_err(|_| {
                ScalixError::ExecutionFailed("Shader source exceeds i32::MAX bytes".to_string())
            })?;
            let ptr = source.as_ptr() as *const c_char;
            (gl.glShaderSource)(shader, 1, &ptr, &len);
            (gl.glCompileShader)(shader);

            let mut success = 0;
            (gl.glGetShaderiv)(shader, GL_COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                (gl.glGetShaderiv)(shader, GL_INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize + 1];
                (gl.glGetShaderInfoLog)(
                    shader,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut c_char,
                );
                (gl.glDeleteShader)(shader);
                let log = String::from_utf8_lossy(&buffer);
                return Err(ScalixError::ExecutionFailed(format!(
                    "Raster shader compilation failed: {log}"
                )));
            }
            Ok(shader)
        }
    }

    fn get_or_build_program(&self, filter: FilterMode) -> Result<RasterProgram> {
        let (target_lock, fs_body) = match filter {
            FilterMode::Area => (&self.prog_area, FRAGMENT_SHADER_AREA_BODY),
            FilterMode::Bicubic => (&self.prog_bicubic, FRAGMENT_SHADER_BICUBIC_BODY),
            FilterMode::Lanczos3 => (&self.prog_lanczos3, FRAGMENT_SHADER_LANCZOS3_BODY),
            _ => (&self.prog_bilinear, FRAGMENT_SHADER_BILINEAR_BODY),
        };

        let mut guard = target_lock.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to lock raster program".to_string())
        })?;
        if let Some(prog) = *guard {
            return Ok(prog);
        }

        let gl = &self.ctx.gl;
        let header = self.ctx.shader_header();
        let vs_src = format!("{header}{VERTEX_SHADER_BODY}");
        let fs_src = format!("{header}{RESAMPLE_KERNELS}{fs_body}");

        let vs = self.compile_shader(GL_VERTEX_SHADER, &vs_src)?;
        let fs = self.compile_shader(GL_FRAGMENT_SHADER, &fs_src)?;

        unsafe {
            let prog = (gl.glCreateProgram)();
            (gl.glAttachShader)(prog, vs);
            (gl.glAttachShader)(prog, fs);
            (gl.glLinkProgram)(prog);

            (gl.glDeleteShader)(vs);
            (gl.glDeleteShader)(fs);

            let mut link_success = 0;
            (gl.glGetProgramiv)(prog, GL_LINK_STATUS, &mut link_success);
            if link_success == 0 {
                let mut len = 0;
                (gl.glGetProgramiv)(prog, GL_INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize + 1];
                (gl.glGetProgramInfoLog)(
                    prog,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut c_char,
                );
                (gl.glDeleteProgram)(prog);
                let log = String::from_utf8_lossy(&buffer);
                return Err(ScalixError::ExecutionFailed(format!(
                    "Raster program link failed: {log}"
                )));
            }

            // The sampler unit never changes, so it is set once here instead of every frame.
            (gl.glUseProgram)(prog);
            let u_tex = (gl.glGetUniformLocation)(prog, c"uTexture".as_ptr() as *const c_char);
            (gl.glUniform1i)(u_tex, 0);
            let program = RasterProgram {
                id: prog,
                u_tex_size: (gl.glGetUniformLocation)(prog, c"uTexSize".as_ptr() as *const c_char),
                u_dst_size: (gl.glGetUniformLocation)(prog, c"uDstSize".as_ptr() as *const c_char),
            };
            *guard = Some(program);
            Ok(program)
        }
    }
    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        let wall_start = Instant::now();
        let _guard = self.ctx.bind_current()?;
        // GL queues work asynchronously; finishing each phase attributes GPU time to it.
        let profiling = self.profiler.is_enabled();
        let gl = &self.ctx.gl;

        let (src_internal, src_format, src_type) =
            get_gl_format_tuple(src.format, self.ctx.is_gles)?;
        let (dst_internal, dst_format, dst_type) =
            get_gl_format_tuple(dst.format, self.ctx.is_gles)?;

        let program = self.get_or_build_program(filter)?;
        let (vao, _vbo) = self.ensure_mesh()?;

        let gl_filter = if filter == FilterMode::Nearest {
            GL_NEAREST
        } else {
            GL_LINEAR
        };

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire GlStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        unsafe {
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);

            // 1. Upload Source from client memory
            let upload_start = Instant::now();
            let src_tex = slot.upload_src_data(
                &self.ctx,
                src,
                src_internal,
                src_format,
                src_type,
                gl_filter,
            )?;

            // 2. Setup Destination FBO
            let dst_tex = slot.ensure_dst_texture(
                &self.ctx,
                dst.width,
                dst.height,
                dst_internal,
                dst_format,
                dst_type,
            )?;

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.dst_fbo);
            (gl.glFramebufferTexture2D)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                dst_tex,
                0,
            );
            if profiling {
                (gl.glFinish)();
            }
            let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

            // 3. Render Quad
            let draw_start = Instant::now();
            (gl.glViewport)(0, 0, dst.width as i32, dst.height as i32);
            (gl.glUseProgram)(program.id);
            if program.u_tex_size >= 0 {
                (gl.glUniform2f)(program.u_tex_size, src.width as f32, src.height as f32);
            }
            if program.u_dst_size >= 0 {
                (gl.glUniform2f)(program.u_dst_size, dst.width as f32, dst.height as f32);
            }

            (gl.glActiveTexture)(GL_TEXTURE0);
            (gl.glBindTexture)(GL_TEXTURE_2D, src_tex);

            (gl.glBindVertexArray)(vao);
            (gl.glDrawArrays)(GL_TRIANGLE_STRIP, 0, 4);
            (gl.glBindVertexArray)(0);

            if profiling {
                (gl.glFinish)();
            }
            let draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;

            // 4. Download Pixels via Persistent FBO
            let download_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.dst_fbo);
            super::ring::read_pixels_into(&self.ctx, dst, dst_format, dst_type, src.format)?;
            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
            let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

            self.ctx.check_gl_error("raster")?;

            let total_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            self.profiler.record(ProfileMetrics {
                host_unpack_ms: 0.0,
                gpu_upload_ms: upload_ms,
                gpu_pure_blit_ms: draw_ms,
                gpu_download_ms: download_ms,
                host_repack_ms: 0.0,
                driver_sync_ms: 0.0,
                total_wall_ms,
            });
        }

        Ok(())
    }
}
