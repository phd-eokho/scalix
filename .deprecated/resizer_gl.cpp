#include "resizer_gl.h"
#include <assert.h>

#include <stdio.h>

using namespace std;

namespace core::shader
{
    const GLchar *vertex_shader = R"glsl(
        #version 300 es
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 texCoords;
        out vec2 TexCoords;

        uniform float imageWidth;
        uniform float imageHeight;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0); // Use vertex positions as-is
            TexCoords = texCoords; // Pass-through texture coordinates
        }

    )glsl";

    const GLchar *fragment_shader_linear = R"glsl(
    #version 300 es
    precision highp float;

    in vec2 TexCoords; // Received from vertex shader
    out vec4 color; // Output color
    uniform sampler2D texture1; // The texture sampler

    uniform float imageWidth;
    uniform float imageHeight;

    void main() {
        vec2 scaledTexCoords = TexCoords;
        float r = imageWidth / imageHeight;
        if(imageWidth < imageHeight) {
            scaledTexCoords.x /= r;

        } else {
            scaledTexCoords.y *= r;
        }

        if(scaledTexCoords.x > 1.0 || scaledTexCoords.y > 1.0) {
            color = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            color = texture(texture1, scaledTexCoords);
        }
    }
    )glsl";

    const GLchar *fragment_shader_bicubic = R"glsl(
    #version 300 es
    precision highp float;

    in vec2 TexCoords; // Received from vertex shader
    out vec4 color; // Output color
    uniform sampler2D texture1; // The texture sampler

    uniform float imageWidth;
    uniform float imageHeight;

    // Cubic interpolation
    float cubic(float x) {
        float a = -0.5; // Adjust 'a' for different bicubic interpolation (e.g., Catmull-Rom with a=-0.5)
        x = abs(x);
        if (x < 1.0) {
            return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
        } else if (x < 2.0) {
            return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
        }
        return 0.0;
    }

    vec4 bicubicSample(vec2 texCoords) {
        vec4 sum = vec4(0.0);
        float weightSum = 0.0;

        float texelWidth = 1.0 / imageWidth;
        float texelHeight = 1.0 / imageHeight;

        for (int m = -1; m <= 2; ++m) {
            for (int n = -1; n <= 2; ++n) {
                vec2 offset = vec2(float(m), float(n)) * vec2(texelWidth, texelHeight);
                float mx = cubic(float(m) - fract(texCoords.x / texelWidth));
                float my = cubic(float(n) - fract(texCoords.y / texelHeight));
                float weight = mx * my;
                sum += texture(texture1, texCoords + offset) * weight;
                weightSum += weight;
            }
        }

        return sum / weightSum;
    }

    void main() {
        vec2 scaledTexCoords = TexCoords;
        float r = imageWidth / imageHeight;
        if(imageWidth < imageHeight) {
            scaledTexCoords.x /= r;
        } else {
            scaledTexCoords.y *= r;
        }

        if(scaledTexCoords.x > 1.0 || scaledTexCoords.y > 1.0) {
            color = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            color = bicubicSample(scaledTexCoords);
        }
    }

)glsl";
}

namespace
{

}

namespace core::image
{
    GLuint ResizerGL::create_program(const GLchar *const *vert, const GLchar *const *frag)
    {
        auto shader_vertex = glCreateShader(GL_VERTEX_SHADER);
        CHECK_GL_ERROR
        glShaderSource(shader_vertex, 1, vert, nullptr);
        CHECK_GL_ERROR
        glCompileShader(shader_vertex);
        CHECK_GL_ERROR

        auto shader_fragment = glCreateShader(GL_FRAGMENT_SHADER);
        CHECK_GL_ERROR
        glShaderSource(shader_fragment, 1, frag, nullptr);
        CHECK_GL_ERROR
        glCompileShader(shader_fragment);
        CHECK_GL_ERROR

        auto program = glCreateProgram();
        CHECK_GL_ERROR
        glAttachShader(program, shader_vertex);
        CHECK_GL_ERROR
        glAttachShader(program, shader_fragment);
        CHECK_GL_ERROR
        glLinkProgram(program);
        CHECK_GL_ERROR

        glDetachShader(program, shader_vertex);
        CHECK_GL_ERROR
        glDeleteShader(shader_vertex);
        CHECK_GL_ERROR

        glDetachShader(program, shader_fragment);
        CHECK_GL_ERROR
        glDeleteShader(shader_fragment);
        CHECK_GL_ERROR

        return program;
    }

    ResizerGL::vao_container ResizerGL::set_vertex()
    {
        // Set up vertex data (and buffer(s)) and configure vertex attributes
        static const float vertices[] = {
            // positions   // texture coords
            1.0f, 1.0f, 1.0f, 1.0f,   // top right
            1.0f, -1.0f, 1.0f, 0.0f,  // bottom right
            -1.0f, -1.0f, 0.0f, 0.0f, // bottom left
            -1.0f, 1.0f, 0.0f, 1.0f   // top left
        };
        static const unsigned int indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
        };

        vao_container container;

        glGenVertexArrays(1, &container.vao);
        CHECK_GL_ERROR

        glGenBuffers(1, &container.vbo);
        CHECK_GL_ERROR
        glGenBuffers(1, &container.ebo);
        CHECK_GL_ERROR

        glBindVertexArray(container.vao);
        CHECK_GL_ERROR

        glBindBuffer(GL_ARRAY_BUFFER, container.vbo);
        CHECK_GL_ERROR
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        CHECK_GL_ERROR

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, container.ebo);
        CHECK_GL_ERROR
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        CHECK_GL_ERROR

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        CHECK_GL_ERROR
        glEnableVertexAttribArray(0);
        CHECK_GL_ERROR

        // Texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
        CHECK_GL_ERROR
        glEnableVertexAttribArray(1);
        CHECK_GL_ERROR

        return container;
    }

    GLuint ResizerGL::create_texture_in()
    {
        GLuint tex;
        glGenTextures(1, &tex);
        CHECK_GL_ERROR
        glBindTexture(GL_TEXTURE_2D, tex);

        auto filter_type = GL_LINEAR;

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_type); // Trilinear filtering
        CHECK_GL_ERROR
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_type); // Linear filtering for magnification
        CHECK_GL_ERROR

        // Set texture wrap mode to clamp to edge
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        CHECK_GL_ERROR
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        CHECK_GL_ERROR

        static GLfloat color[] = {0.0f, 0.0f, 0.0f, 1.0f}; // Black border
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
        CHECK_GL_ERROR

        // Check for and apply anisotropic filtering if supported
        GLfloat max_anisotropy;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy);
        CHECK_GL_ERROR

        GLfloat anisotropy = std::max(4.0f, max_anisotropy); // Or use maxAnisotropy for maximum quality
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
        CHECK_GL_ERROR

        return tex;
    }

    GLuint ResizerGL::create_fbo()
    {
        GLuint FBO;
        glGenFramebuffers(1, &FBO);
        CHECK_GL_ERROR
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        CHECK_GL_ERROR
        return FBO;
    }

    GLuint ResizerGL::create_texture_out(int width, int height)
    {
        GLuint tex;

        glGenTextures(1, &tex);
        CHECK_GL_ERROR
        glBindTexture(GL_TEXTURE_2D, tex);
        CHECK_GL_ERROR
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        CHECK_GL_ERROR

        return tex;
    }

    void ResizerGL::deleter_program(GLuint *p)
    {
        glDeleteProgram(*p);
        CHECK_GL_ERROR
    }

    void ResizerGL::deleter_vao(vao_container *p)
    {
        glBindVertexArray(0);
        CHECK_GL_ERROR

        glDeleteVertexArrays(1, &p->vao);
        CHECK_GL_ERROR

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CHECK_GL_ERROR
        glDeleteBuffers(1, &p->vbo);
        CHECK_GL_ERROR

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        CHECK_GL_ERROR
        glDeleteBuffers(1, &p->ebo);
        CHECK_GL_ERROR
    }

    void ResizerGL::deleter_texture(GLuint *p)
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        CHECK_GL_ERROR

        glDeleteTextures(1, p);
        CHECK_GL_ERROR
    }

    void ResizerGL::deleter_fbo(GLuint *p)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        CHECK_GL_ERROR
        glDeleteFramebuffers(1, p);
        CHECK_GL_ERROR
    }

    ResizerGL::ResizerGL(int width, int height)
        : width(width), height(height),
          program(0), p_program(&program, deleter_program),
          vao{0, 0, 0}, p_vao(&vao, deleter_vao),
          texture_in(0), p_texture_in(&texture_in, deleter_texture),
          texture_out(0), p_texture_out(&texture_out, deleter_texture),
          fbo(0), p_fbo(&fbo, deleter_fbo)
    {
    }

    ResizerGL::~ResizerGL()
    {
        p_program.reset();

        p_fbo.reset();

        p_texture_in.reset();
        p_texture_out.reset();

        p_vao.reset();
    }

    void ResizerGL::init(int width, int height)
    {
        program = create_program(&shader::vertex_shader, &shader::fragment_shader_bicubic);
        assert(program != 0);

        vao = set_vertex();
        assert(vao.vao != 0);
        assert(vao.vbo != 0);
        assert(vao.ebo != 0);

        texture_in = create_texture_in();
        assert(texture_in != 0);

        fbo = create_fbo();
        assert(fbo != 0);

        texture_out = create_texture_out(width, height);
        assert(texture_out != 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_out, 0);
        CHECK_GL_ERROR
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

        glViewport(0, 0, width, height);
        CHECK_GL_ERROR

        glUseProgram(program);
        CHECK_GL_ERROR

        auto tex1 = glGetUniformLocation(program, "texture1");
        CHECK_GL_ERROR
        glUniform1i(tex1, 0);
        CHECK_GL_ERROR

        tex_w = glGetUniformLocation(program, "imageWidth");
        CHECK_GL_ERROR
        assert(tex_w >= 0);

        tex_h = glGetUniformLocation(program, "imageHeight");
        CHECK_GL_ERROR
        assert(tex_h >= 0);
    }

    void ResizerGL::resize(uint8_t *ptr, const uint8_t *im, int w, int h)
    {
        assert(ptr != nullptr && im != nullptr);

        glBindTexture(GL_TEXTURE_2D, texture_in);
        CHECK_GL_ERROR
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, im);
        CHECK_GL_ERROR

        // glGenerateMipmap(GL_TEXTURE_2D);
        // CHECK_GL_ERROR

        glUniform1f(tex_w, w);
        CHECK_GL_ERROR

        glUniform1f(tex_h, h);
        CHECK_GL_ERROR

        glBindVertexArray(vao.vao);
        CHECK_GL_ERROR

        glActiveTexture(GL_TEXTURE0);
        CHECK_GL_ERROR
        glBindTexture(GL_TEXTURE_2D, texture_in);
        CHECK_GL_ERROR

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        CHECK_GL_ERROR

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        CHECK_GL_ERROR

        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, ptr);
        CHECK_GL_ERROR
    }

    void ResizerEGL::deleter_display(EGLDisplay *p)
    {
        auto ret = eglTerminate(*p);
        assert(ret == EGL_TRUE);
    }

    void ResizerEGL::deleter_context(context_container *p)
    {
        auto ret = eglDestroyContext(p->display, p->ctx);
        assert(ret == EGL_TRUE);
    }

    void ResizerEGL::deleter_surface(surface_container *p)
    {
        auto ret = eglDestroySurface(p->display, p->surf);
        assert(ret == EGL_TRUE);
    }

    ResizerEGL::ResizerEGL(int width, int height)
        : width(width), height(height), display(EGL_NO_DISPLAY), p_display(&display, deleter_display),
          context(display), p_context(&context, deleter_context), surface(display), p_surface(&surface, deleter_surface),
          resizer(make_unique<ResizerGL>(width, height))
    {
        init(width, height);
    }

    ResizerEGL::~ResizerEGL()
    {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

        resizer.reset();
        glFinish();

        p_surface.reset();
        p_context.reset();
        p_display.reset();
    }

    void ResizerEGL::init(int width, int height)
    {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        assert(display != EGL_NO_DISPLAY);

        int32_t v_major, v_minor;
        eglInitialize(display, &v_major, &v_minor);
        LOG("EGL version: %d.%d\n", v_major, v_minor);

        const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_DEPTH_SIZE, 8,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_NONE};

        EGLConfig eglConfig;
        EGLint numConfigs;
        eglChooseConfig(display, configAttribs, &eglConfig, 1, &numConfigs);

        const EGLint contextAttribs[] = {
            EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL_NONE};

        context.ctx = eglCreateContext(display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
        assert(context.ctx != EGL_NO_CONTEXT);

        const EGLint pbufferAttribs[] = {
            EGL_WIDTH,
            width,
            EGL_HEIGHT,
            height,
            EGL_NONE,
        };
        surface.surf = eglCreatePbufferSurface(display, eglConfig, pbufferAttribs);

        eglMakeCurrent(display, surface.surf, surface.surf, context.ctx);

        resizer->init(width, height);
    }
}