#pragma once

#include <scalix/scalix.h>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

namespace scalix {

enum class Backend : int {
    Auto        = SCALIX_BACKEND_AUTO,
    Vulkan      = SCALIX_BACKEND_VULKAN,
    OpenGL      = SCALIX_BACKEND_OPENGL,
    Npu         = SCALIX_BACKEND_NPU,
    Hw2d        = SCALIX_BACKEND_HW2D,
    Cpu         = SCALIX_BACKEND_CPU,
    Passthrough = SCALIX_BACKEND_PASSTHROUGH
};

enum class Filter : int {
    Nearest     = SCALIX_FILTER_NEAREST,
    Bilinear    = SCALIX_FILTER_BILINEAR,
    Bicubic     = SCALIX_FILTER_BICUBIC,
    Lanczos3    = SCALIX_FILTER_LANCZOS3,
    Area        = SCALIX_FILTER_AREA,
    Passthrough = SCALIX_FILTER_PASSTHROUGH
};

enum class PixelFormat : int {
    Rgba8888 = SCALIX_FORMAT_RGBA8888,
    Bgra8888 = SCALIX_FORMAT_BGRA8888,
    Rgb888   = SCALIX_FORMAT_RGB888,
    Bgr888   = SCALIX_FORMAT_BGR888,
    R8       = SCALIX_FORMAT_R8,
    Rg88     = SCALIX_FORMAT_RG88,
    Nv12     = SCALIX_FORMAT_NV12,
    Yuv420p  = SCALIX_FORMAT_YUV420P,
    Rgba16f  = SCALIX_FORMAT_RGBA16F,
    Rgba32f  = SCALIX_FORMAT_RGBA32F
};

struct ImageDesc {
    uint32_t width{0};
    uint32_t height{0};
    size_t stride_bytes{0};
    PixelFormat format{PixelFormat::Rgba8888};
    uint8_t* host_ptr{nullptr};
    size_t data_len{0};
    int dma_buf_fd{-1};

    ScalixImageDesc to_c() const {
        return ScalixImageDesc{
            .width = width,
            .height = height,
            .stride_bytes = stride_bytes,
            .format = static_cast<ScalixPixelFormat>(format),
            .host_ptr = host_ptr,
            .data_len = data_len,
            .dma_buf_fd = dma_buf_fd,
        };
    }
};

class Engine {
public:
    /**
     * @brief Constructs a new Scalix Engine instance.
     * 
     * @param backend Hardware accelerator backend type (default: Backend::Auto).
     * @param thread_prefix Optional custom naming prefix for engine worker threads
     *                      (e.g., `<prefix>/scx-hw` and `<prefix>/scx-w<id>`). If nullptr
     *                      or empty, defaults to current process ID (`<pid>`).
     * @note Maximum effective length of @p thread_prefix is 7 characters.
     * @warning Prefixes longer than 7 characters are automatically truncated to 7 characters
     *          to guarantee strict compliance with Linux OS 15-character thread name (`comm`) limit.
     */
    explicit Engine(Backend backend = Backend::Auto, const char* thread_prefix = nullptr) {
        engine_ = scalix_engine_create_with_prefix(
            static_cast<ScalixBackendType>(backend),
            thread_prefix
        );
        if (!engine_) {
            throw std::runtime_error("Failed to create Scalix Engine");
        }
    }

    ~Engine() {
        if (engine_) {
            scalix_engine_destroy(engine_);
            engine_ = nullptr;
        }
    }

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&& other) noexcept : engine_(other.engine_) {
        other.engine_ = nullptr;
    }

    Engine& operator=(Engine&& other) noexcept {
        if (this != &other) {
            if (engine_) {
                scalix_engine_destroy(engine_);
            }
            engine_ = other.engine_;
            other.engine_ = nullptr;
        }
        return *this;
    }

    /// Synchronous execution
    void resize(const ImageDesc& src, ImageDesc& dst, Filter filter = Filter::Passthrough) {
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();
        int status = scalix_resize_sync(
            engine_,
            &c_src,
            &c_dst,
            static_cast<ScalixFilterMode>(filter)
        );
        if (status != SCALIX_SUCCESS) {
            throw std::runtime_error("Scalix resize_sync failed with status code: " + std::to_string(status));
        }
    }

    /// Callback-driven execution
    void resize_callback(
        const ImageDesc& src,
        const ImageDesc& dst,
        Filter filter,
        std::function<void(int status)> callback
    ) {
        auto* cb_ptr = new std::function<void(int status)>(std::move(callback));
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();

        int status = scalix_resize_submit(
            engine_,
            &c_src,
            &c_dst,
            static_cast<ScalixFilterMode>(filter),
            [](int code, void* user_data) {
                auto* cb = static_cast<std::function<void(int status)>*>(user_data);
                if (cb) {
                    (*cb)(code);
                    delete cb;
                }
            },
            cb_ptr
        );

        if (status != SCALIX_SUCCESS) {
            delete cb_ptr;
            throw std::runtime_error("Scalix resize_submit failed with status code: " + std::to_string(status));
        }
    }

private:
    ScalixEngine* engine_{nullptr};
};

} // namespace scalix
