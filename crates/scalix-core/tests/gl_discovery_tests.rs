use scalix_core::{GlBackend, ScalixError};

const GL_RENDERER: u32 = 0x1F01;

/// Mesa's software EGL device (llvmpipe) must not win over a hardware-capable display.
/// Only meaningful on hosts with a GPU driver; with `LIBGL_ALWAYS_SOFTWARE=1` (CI) or no GPU,
/// llvmpipe is the expected outcome and the test only reports it.
#[test]
fn test_gl_prefers_hardware_renderer() {
    let backend = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let ctx = backend.context();
    let _guard = ctx.bind_current().expect("bind EGL context");
    let renderer = unsafe {
        let p = (ctx.gl.glGetString)(GL_RENDERER);
        assert!(!p.is_null(), "GL_RENDERER is null");
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    };
    eprintln!("GL_RENDERER = {renderer}");

    let software_forced = std::env::var_os("LIBGL_ALWAYS_SOFTWARE").is_some();
    // NOTE: WSL2 (/dev/dxg) is excluded: Mesa only exposes D3D12 to headless EGL with
    // GALLIUM_DRIVER=d3d12. Action: set it when benchmarking GL on WSL.
    let gpu_present = std::fs::read_dir("/dev/dri")
        .map(|d| {
            d.flatten()
                .any(|e| e.file_name().to_string_lossy().starts_with("renderD"))
        })
        .unwrap_or(false);
    if gpu_present && !software_forced {
        assert!(
            !renderer.contains("llvmpipe") && !renderer.contains("softpipe"),
            "GPU present but software renderer selected: {renderer}"
        );
    }
}
