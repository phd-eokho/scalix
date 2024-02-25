#pragma once

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include <memory>

#define CHECK_GL_ERROR                                           \
    if (auto err = glGetError(); err != GL_NO_ERROR)             \
    {                                                            \
        printf("%s:%d GL error: %d\n", __FILE__, __LINE__, err); \
    }

#define LOG(msg, ...) printf(msg, __VA_ARGS__)

namespace core::image
{
    class ResizerGL
    {
    private:
        GLuint create_program(const GLchar *const *vert, const GLchar *const *frag);

        GLuint create_texture_in();

        GLuint create_fbo();

        GLuint create_texture_out(int width, int height);

    protected:
        const int32_t width;
        const int32_t height;

        GLuint program;
        static void deleter_program(GLuint *p);
        std::unique_ptr<GLuint, void (*)(GLuint *p)> p_program;

        struct vao_container
        {
            GLuint vao;
            GLuint vbo;
            GLuint ebo;
        } vao;
        static void deleter_vao(vao_container *p);
        std::unique_ptr<vao_container, void (*)(vao_container *p)> p_vao;

        static void deleter_texture(GLuint *p);

        GLuint texture_in;
        std::unique_ptr<GLuint, void (*)(GLuint *p)> p_texture_in;
        GLuint texture_out;
        std::unique_ptr<GLuint, void (*)(GLuint *p)> p_texture_out;

        GLuint fbo;
        static void deleter_fbo(GLuint *p);
        std::unique_ptr<GLuint, void (*)(GLuint *p)> p_fbo;

        GLint tex_w;
        GLint tex_h;

    private:
        vao_container set_vertex();

    public:
        ResizerGL(int width, int height);
        virtual ~ResizerGL();

        virtual void init(int width, int height);
        virtual void resize(uint8_t *ptr, const uint8_t *im, int w, int h);
    };
    class ResizerEGL
    {
    protected:
        const int32_t width;
        const int32_t height;

        EGLDisplay display;
        static void deleter_display(EGLDisplay *p);
        std::unique_ptr<EGLDisplay, void (*)(EGLDisplay *)> p_display;

        struct context_container
        {
            const EGLDisplay &display;
            EGLContext ctx = EGL_NO_CONTEXT;
            context_container(EGLDisplay &_disp) : display(_disp) {}
        } context;
        static void deleter_context(context_container *p);
        std::unique_ptr<context_container, void (*)(context_container *)> p_context;

        struct surface_container
        {
            const EGLDisplay &display;
            EGLSurface surf = EGL_NO_SURFACE;
            surface_container(EGLDisplay &_disp) : display(_disp) {}
        } surface;
        static void deleter_surface(surface_container *p);
        std::unique_ptr<surface_container, void (*)(surface_container *)> p_surface;

        std::unique_ptr<ResizerGL> resizer;

        virtual void init(int width, int height);

    public:
        ResizerEGL(int width, int height);
        virtual ~ResizerEGL();

        virtual void resize(uint8_t *ptr, const uint8_t *im, int w, int h)
        {
            eglMakeCurrent(display, surface.surf, surface.surf, context.ctx);
            resizer->resize(ptr, im, w, h);
        }
    };
}