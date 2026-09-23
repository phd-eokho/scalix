//! OpenGL Staging Ring Buffer & Pipeline Resource Manager
//!
//! Provides triple-buffered PBO (Pixel Buffer Object) staging rings, persistent
//! 2D texture pooling, and per-slot `GLsync` fence synchronization to eliminate
//! per-frame allocations and enable seamless CPU/GPU asynchronous pipelining.

use std::sync::Arc;
use std::ffi::c_void;
use crate::backend::gl::context::*;
use crate::types::Result;

pub const DEFAULT_GL_RING_SLOTS: usize = 3;

/// An isolated in-flight execution slot in the OpenGL staging ring.
pub struct GlStagingSlot {
    /// Pixel Unpack Buffer (CPU -> GPU asynchronous upload)
    pub src_pbo: u32,
    pub src_pbo_size: usize,

    /// Pixel Pack Buffer (GPU -> CPU asynchronous download)
    pub dst_pbo: u32,
    pub dst_pbo_size: usize,

    /// Cached persistent source texture (reused across frames)
    pub src_texture: u32,
    pub src_tex_w: u32,
    pub src_tex_h: u32,
    pub src_tex_internal: u32,
    pub src_tex_format: u32,
    pub src_tex_type: u32,

    /// Cached persistent destination texture (reused across frames)
    pub dst_texture: u32,
    pub dst_tex_w: u32,
    pub dst_tex_h: u32,
    pub dst_tex_internal: u32,
    pub dst_tex_format: u32,
    pub dst_tex_type: u32,

    /// Framebuffer Objects for Blit/Raster/ReadPixels
    pub src_fbo: u32,
    pub dst_fbo: u32,

    /// Hardware completion sync fence
    pub sync_fence: Option<GLsync>,
    pub in_flight: bool,
}

impl GlStagingSlot {
    pub fn new(ctx: &Arc<EglContext>) -> Result<Self> {
        let gl = &ctx.gl;
        let mut pbos = [0u32; 2];
        let mut textures = [0u32; 2];
        let mut fbos = [0u32; 2];

        unsafe {
            (gl.glGenBuffers)(2, pbos.as_mut_ptr());
            (gl.glGenTextures)(2, textures.as_mut_ptr());
            (gl.glGenFramebuffers)(2, fbos.as_mut_ptr());
        }

        Ok(Self {
            src_pbo: pbos[0],
            src_pbo_size: 0,
            dst_pbo: pbos[1],
            dst_pbo_size: 0,
            src_texture: textures[0],
            src_tex_w: 0,
            src_tex_h: 0,
            src_tex_internal: 0,
            src_tex_format: 0,
            src_tex_type: 0,
            dst_texture: textures[1],
            dst_tex_w: 0,
            dst_tex_h: 0,
            dst_tex_internal: 0,
            dst_tex_format: 0,
            dst_tex_type: 0,
            src_fbo: fbos[0],
            dst_fbo: fbos[1],
            sync_fence: None,
            in_flight: false,
        })
    }

    /// Ensures that the slot has a source PBO with at least `min_size` bytes.
    ///
    /// Reuses the existing buffer if `src_pbo_size >= min_size`.
    /// Preserves stability via hysteresis shrinking if `min_size < src_pbo_size / 2`.
    pub fn ensure_src_pbo(&mut self, ctx: &Arc<EglContext>, min_size: usize) -> Result<u32> {
        let needs_realloc = self.src_pbo_size < min_size || min_size < self.src_pbo_size / 2;
        if needs_realloc {
            let gl = &ctx.gl;
            let new_size = min_size.max(64 * 1024);
            unsafe {
                (gl.glBindBuffer)(GL_PIXEL_UNPACK_BUFFER, self.src_pbo);
                (gl.glBufferData)(GL_PIXEL_UNPACK_BUFFER, new_size as isize, std::ptr::null(), GL_STREAM_DRAW);
                (gl.glBindBuffer)(GL_PIXEL_UNPACK_BUFFER, 0);
            }
            self.src_pbo_size = new_size;
        }
        Ok(self.src_pbo)
    }

    /// Ensures that the slot has a destination PBO with at least `min_size` bytes.
    pub fn ensure_dst_pbo(&mut self, ctx: &Arc<EglContext>, min_size: usize) -> Result<u32> {
        let needs_realloc = self.dst_pbo_size < min_size || min_size < self.dst_pbo_size / 2;
        if needs_realloc {
            let gl = &ctx.gl;
            let new_size = min_size.max(64 * 1024);
            unsafe {
                (gl.glBindBuffer)(GL_PIXEL_PACK_BUFFER, self.dst_pbo);
                (gl.glBufferData)(GL_PIXEL_PACK_BUFFER, new_size as isize, std::ptr::null(), GL_STREAM_READ);
                (gl.glBindBuffer)(GL_PIXEL_PACK_BUFFER, 0);
            }
            self.dst_pbo_size = new_size;
        }
        Ok(self.dst_pbo)
    }

    /// Ensures source texture allocation is configured with matching dimensions and format.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_src_texture(
        &mut self,
        ctx: &Arc<EglContext>,
        w: u32,
        h: u32,
        internal: u32,
        format: u32,
        typ: u32,
        gl_filter: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        let needs_realloc = self.src_tex_w != w
            || self.src_tex_h != h
            || self.src_tex_internal != internal
            || self.src_tex_format != format
            || self.src_tex_type != typ;

        unsafe {
            (gl.glBindTexture)(GL_TEXTURE_2D, self.src_texture);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gl_filter as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gl_filter as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);

            if needs_realloc {
                (gl.glTexImage2D)(
                    GL_TEXTURE_2D, 0, internal as i32,
                    w as i32, h as i32, 0,
                    format, typ, std::ptr::null(),
                );
                self.src_tex_w = w;
                self.src_tex_h = h;
                self.src_tex_internal = internal;
                self.src_tex_format = format;
                self.src_tex_type = typ;
            }
        }
        Ok(self.src_texture)
    }

    /// Ensures destination texture allocation is configured with matching dimensions and format.
    pub fn ensure_dst_texture(
        &mut self,
        ctx: &Arc<EglContext>,
        w: u32,
        h: u32,
        internal: u32,
        format: u32,
        typ: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        let needs_realloc = self.dst_tex_w != w
            || self.dst_tex_h != h
            || self.dst_tex_internal != internal
            || self.dst_tex_format != format
            || self.dst_tex_type != typ;

        unsafe {
            (gl.glBindTexture)(GL_TEXTURE_2D, self.dst_texture);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);

            if needs_realloc {
                (gl.glTexImage2D)(
                    GL_TEXTURE_2D, 0, internal as i32,
                    w as i32, h as i32, 0,
                    format, typ, std::ptr::null(),
                );
                self.dst_tex_w = w;
                self.dst_tex_h = h;
                self.dst_tex_internal = internal;
                self.dst_tex_format = format;
                self.dst_tex_type = typ;
            }
        }
        Ok(self.dst_texture)
    }

    /// Uploads source pixels via PBO DMA staging without CPU blocking.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_src_data(
        &mut self,
        ctx: &Arc<EglContext>,
        data: &[u8],
        w: u32,
        h: u32,
        internal: u32,
        format: u32,
        typ: u32,
        gl_filter: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        self.ensure_src_pbo(ctx, data.len())?;
        self.ensure_src_texture(ctx, w, h, internal, format, typ, gl_filter)?;

        unsafe {
            (gl.glBindBuffer)(GL_PIXEL_UNPACK_BUFFER, self.src_pbo);
            (gl.glBufferData)(
                GL_PIXEL_UNPACK_BUFFER,
                data.len() as isize,
                data.as_ptr() as *const c_void,
                GL_STREAM_DRAW,
            );

            (gl.glBindTexture)(GL_TEXTURE_2D, self.src_texture);
            (gl.glTexSubImage2D)(
                GL_TEXTURE_2D, 0, 0, 0,
                w as i32, h as i32,
                format, typ, std::ptr::null(),
            );
            (gl.glBindBuffer)(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        Ok(self.src_texture)
    }

    /// Waits for any in-flight GPU workload on this slot and resets the fence.
    pub fn wait_and_reset(&mut self, ctx: &Arc<EglContext>) -> Result<()> {
        if self.in_flight {
            let gl = &ctx.gl;
            if let Some(fence) = self.sync_fence.take() {
                if let Some(gl_wait) = gl.glClientWaitSync {
                    unsafe {
                        let res = gl_wait(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
                        if res == GL_WAIT_FAILED {
                            log::warn!("glClientWaitSync returned GL_WAIT_FAILED");
                        }
                    }
                }
                if let Some(gl_del) = gl.glDeleteSync {
                    unsafe {
                        gl_del(fence);
                    }
                }
            } else {
                unsafe {
                    (gl.glFinish)();
                }
            }
            self.in_flight = false;
        }
        Ok(())
    }

    /// Signals a GPU fence on this slot.
    pub fn signal_fence(&mut self, ctx: &Arc<EglContext>) {
        let gl = &ctx.gl;
        if let Some(gl_fence) = gl.glFenceSync {
            let fence = unsafe { gl_fence(GL_SYNC_GPU_COMMANDS_COMPLETE, 0) };
            if !fence.is_null() {
                self.sync_fence = Some(fence);
            }
        }
        unsafe {
            (gl.glFlush)();
        }
        self.in_flight = true;
    }
}

unsafe impl Send for GlStagingSlot {}
unsafe impl Sync for GlStagingSlot {}

/// Ring-buffered staging allocator and execution coordinator for OpenGL/GLES.
pub struct GlStagingRing {
    ctx: Arc<EglContext>,
    slots: Vec<GlStagingSlot>,
    current_idx: usize,
}

unsafe impl Send for GlStagingRing {}
unsafe impl Sync for GlStagingRing {}

impl GlStagingRing {
    /// Creates a new OpenGL staging ring with `slot_count` slots (default: 3).
    pub fn new(ctx: Arc<EglContext>, slot_count: usize) -> Result<Self> {
        let count = slot_count.max(1);
        let mut slots = Vec::with_capacity(count);

        {
            let _guard = ctx.bind_current()?;
            for _ in 0..count {
                slots.push(GlStagingSlot::new(&ctx)?);
            }
        }

        Ok(Self {
            ctx,
            slots,
            current_idx: 0,
        })
    }

    /// Acquires the next available slot in the ring, waiting for prior GPU execution to complete.
    pub fn acquire_slot(&mut self) -> Result<(usize, &mut GlStagingSlot)> {
        let idx = self.current_idx;
        self.current_idx = (self.current_idx + 1) % self.slots.len();

        let slot = &mut self.slots[idx];
        slot.wait_and_reset(&self.ctx)?;
        Ok((idx, slot))
    }
}

impl Drop for GlStagingRing {
    fn drop(&mut self) {
        if let Ok(_guard) = self.ctx.bind_current() {
            let gl = &self.ctx.gl;
            for slot in &mut self.slots {
                let _ = slot.wait_and_reset(&self.ctx);
                unsafe {
                    let pbos = [slot.src_pbo, slot.dst_pbo];
                    let textures = [slot.src_texture, slot.dst_texture];
                    let fbos = [slot.src_fbo, slot.dst_fbo];

                    (gl.glDeleteBuffers)(2, pbos.as_ptr());
                    (gl.glDeleteTextures)(2, textures.as_ptr());
                    (gl.glDeleteFramebuffers)(2, fbos.as_ptr());
                }
            }
        }
    }
}
