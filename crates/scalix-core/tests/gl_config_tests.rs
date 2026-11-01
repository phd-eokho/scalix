use scalix_core::{GlBackend, ScalixError};

/// Desktop Linux drivers (Mesa, NVIDIA) support desktop GL; the backend must not silently fall
/// back to GLES because the chosen EGL config lacked EGL_OPENGL_BIT.
#[cfg(all(target_os = "linux", not(target_os = "android")))]
#[test]
fn test_gl_desktop_context_on_linux() {
    let backend = match GlBackend::new() {
        Ok(b) => b,
        Err(ScalixError::BackendUnavailable(_)) => {
            eprintln!("OpenGL/EGL backend not available on this host; skipping");
            return;
        }
        Err(e) => panic!("Unexpected error initializing GlBackend: {e:?}"),
    };
    let version = &backend.context().gl_version;
    eprintln!("GL_VERSION = {version}");
    assert!(
        !version.starts_with("OpenGL ES"),
        "expected a desktop GL context on Linux, got '{version}'"
    );
}
