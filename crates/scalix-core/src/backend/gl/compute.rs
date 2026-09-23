//! OpenGL Compute Shader Resizer (`glDispatchCompute`)

use std::ffi::{c_char, CString};
use std::sync::Arc;
use std::time::Instant;

use super::blit::get_gl_format_tuple;
use super::context::*;
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

const COMPUTE_SHADER_BODY: &str = r#"
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D uSrcTexture;
layout(binding = 1, rgba8) uniform writeonly highp image2D uDstImage;

void main() {
    ivec2 dstCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dstSize = imageSize(uDstImage);
    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y) {
        return;
    }

    vec2 uv = (vec2(dstCoord) + 0.5) / vec2(dstSize);
    vec4 color = texture(uSrcTexture, uv);
    imageStore(uDstImage, dstCoord, color);
}
"#;

use std::sync::Mutex;
use super::ring::GlStagingRing;

pub struct GlComputeResizer {
    ctx: Arc<EglContext>,
    ring: Arc<Mutex<GlStagingRing>>,
    profiler: Arc<dyn Profiler>,
    program: Mutex<Option<u32>>,
}

impl GlComputeResizer {
    pub fn new(
        ctx: Arc<EglContext>,
        ring: Arc<Mutex<GlStagingRing>>,
        profiler: Arc<dyn Profiler>,
    ) -> Self {
        Self {
            ctx,
            ring,
            profiler,
            program: Mutex::new(None),
        }
    }

    pub fn is_supported(&self) -> bool {
        self.ctx.is_compute_supported
    }

    fn get_or_build_program(&self) -> Result<u32> {
        let mut prog_lock = self.program.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire compute program lock".to_string())
        })?;
        if let Some(prog) = *prog_lock {
            return Ok(prog);
        }
        let prog = self.build_program()?;
        *prog_lock = Some(prog);
        Ok(prog)
    }

    fn build_program(&self) -> Result<u32> {
        let gl = &self.ctx.gl;
        let header = self.ctx.compute_shader_header();
        let src = format!("{header}{COMPUTE_SHADER_BODY}");

        unsafe {
            let shader = (gl.glCreateShader)(GL_COMPUTE_SHADER);
            let c_src = CString::new(src).unwrap();
            let ptr = c_src.as_ptr();
            (gl.glShaderSource)(shader, 1, &ptr, std::ptr::null());
            (gl.glCompileShader)(shader);

            let mut success = 0;
            (gl.glGetShaderiv)(shader, GL_COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                (gl.glGetShaderiv)(shader, GL_INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize + 1];
                (gl.glGetShaderInfoLog)(shader, len, std::ptr::null_mut(), buffer.as_mut_ptr() as *mut c_char);
                (gl.glDeleteShader)(shader);
                let log = String::from_utf8_lossy(&buffer);
                return Err(ScalixError::ExecutionFailed(format!("Compute shader compilation failed: {log}")));
            }

            let prog = (gl.glCreateProgram)();
            (gl.glAttachShader)(prog, shader);
            (gl.glLinkProgram)(prog);
            (gl.glDeleteShader)(shader);

            let mut link_success = 0;
            (gl.glGetProgramiv)(prog, GL_LINK_STATUS, &mut link_success);
            if link_success == 0 {
                let mut len = 0;
                (gl.glGetProgramiv)(prog, GL_INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize + 1];
                (gl.glGetProgramInfoLog)(prog, len, std::ptr::null_mut(), buffer.as_mut_ptr() as *mut c_char);
                (gl.glDeleteProgram)(prog);
                let log = String::from_utf8_lossy(&buffer);
                return Err(ScalixError::ExecutionFailed(format!("Compute program link failed: {log}")));
            }
            Ok(prog)
        }
    }

    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        if !self.is_supported() {
            return Err(ScalixError::ExecutionFailed(
                "OpenGL Compute Shaders (glDispatchCompute) are not supported on this device".to_string(),
            ));
        }

        let wall_start = Instant::now();
        let _guard = self.ctx.bind_current()?;
        let gl = &self.ctx.gl;

        let gl_dispatch = gl.glDispatchCompute.ok_or_else(|| {
            ScalixError::ExecutionFailed("glDispatchCompute not available".to_string())
        })?;
        let gl_bind_image = gl.glBindImageTexture.ok_or_else(|| {
            ScalixError::ExecutionFailed("glBindImageTexture not available".to_string())
        })?;

        let (src_internal, src_format, src_type) = get_gl_format_tuple(src.format)?;
        let (_dst_internal, dst_format, dst_type) = get_gl_format_tuple(dst.format)?;

        let program = self.get_or_build_program()?;
        let gl_filter = if filter == FilterMode::Nearest { GL_NEAREST } else { GL_LINEAR };

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire GlStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        unsafe {
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);

            // 1. Upload Source via PBO DMA staging
            let upload_start = Instant::now();
            let src_tex = slot.upload_src_data(
                &self.ctx,
                src.data,
                src.width,
                src.height,
                src_internal,
                src_format,
                src_type,
                gl_filter,
            )?;
            let dst_tex = slot.ensure_dst_texture(
                &self.ctx,
                dst.width,
                dst.height,
                GL_RGBA8,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
            )?;
            let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

            // 2. Dispatch Compute
            let compute_start = Instant::now();
            (gl.glUseProgram)(program);

            (gl.glActiveTexture)(GL_TEXTURE0);
            (gl.glBindTexture)(GL_TEXTURE_2D, src_tex);
            let u_tex = (gl.glGetUniformLocation)(program, c"uSrcTexture".as_ptr() as *const c_char);
            (gl.glUniform1i)(u_tex, 0);

            gl_bind_image(1, dst_tex, 0, 0, 0, GL_WRITE_ONLY, GL_RGBA8);

            let groups_x = dst.width.div_ceil(16);
            let groups_y = dst.height.div_ceil(16);
            gl_dispatch(groups_x, groups_y, 1);

            if let Some(gl_barrier) = gl.glMemoryBarrier {
                gl_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ALL_BARRIER_BITS);
            }
            slot.signal_fence(&self.ctx);
            let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

            // 3. Download Pixels via Persistent FBO
            let download_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.dst_fbo);
            (gl.glFramebufferTexture2D)(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);

            (gl.glReadPixels)(
                0, 0, dst.width as i32, dst.height as i32,
                dst_format, dst_type, dst.data.as_mut_ptr() as *mut std::ffi::c_void,
            );
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, 0);
            let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

            let total_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            self.profiler.record(ProfileMetrics {
                host_unpack_ms: 0.0,
                gpu_upload_ms: upload_ms,
                gpu_pure_blit_ms: compute_ms,
                gpu_download_ms: download_ms,
                host_repack_ms: 0.0,
                driver_sync_ms: 0.0,
                total_wall_ms,
            });
        }

        Ok(())
    }
}
