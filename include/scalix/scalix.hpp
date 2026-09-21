#pragma once

#include <scalix/scalix.h>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace scalix {

using ProfileMetrics = ScalixProfileMetrics;

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

/**
 * @brief RAII wrapper for a hardware-backed zero-copy DMA buffer.
 */
class DmaBuffer {
public:
    /**
     * @brief Allocates a new DMA buffer.
     * 
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param format Pixel format.
     */
    DmaBuffer(uint32_t width, uint32_t height, PixelFormat format = PixelFormat::Rgba8888) {
        handle_ = scalix_dma_buffer_allocate(
            width,
            height,
            static_cast<ScalixPixelFormat>(format)
        );
        if (!handle_) {
            throw std::runtime_error("Failed to allocate DMA buffer (DMA-Heap/DRM unavailable)");
        }
    }

    ~DmaBuffer() {
        if (handle_) {
            scalix_dma_buffer_free(handle_);
            handle_ = nullptr;
        }
    }

    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    DmaBuffer(DmaBuffer&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    DmaBuffer& operator=(DmaBuffer&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                scalix_dma_buffer_free(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    int fd() const {
        return handle_ ? scalix_dma_buffer_get_fd(handle_) : -1;
    }

    uint8_t* host_ptr() const {
        return handle_ ? scalix_dma_buffer_get_host_ptr(handle_) : nullptr;
    }

    size_t size() const {
        return handle_ ? scalix_dma_buffer_get_size(handle_) : 0;
    }

    size_t stride() const {
        return handle_ ? scalix_dma_buffer_get_stride(handle_) : 0;
    }

    void sync_start(bool is_write = true) {
        if (handle_) {
            int ret = scalix_dma_buffer_sync_start(handle_, is_write);
            if (ret != SCALIX_SUCCESS) {
                throw std::runtime_error("scalix_dma_buffer_sync_start failed: " + std::to_string(ret));
            }
        }
    }

    void sync_end(bool is_write = true) {
        if (handle_) {
            int ret = scalix_dma_buffer_sync_end(handle_, is_write);
            if (ret != SCALIX_SUCCESS) {
                throw std::runtime_error("scalix_dma_buffer_sync_end failed: " + std::to_string(ret));
            }
        }
    }

    /**
     * @brief Executes a callable (e.g. lambda) with automatic CPU write synchronization.
     * 
     * Automatically calls `sync_start(true)` before invocation and guarantees `sync_end(true)`
     * on scope exit via RAII (even if an exception occurs).
     * 
     * @tparam Func Callable accepting `(uint8_t* host_ptr, size_t size, Args...)` or `(uint8_t* host_ptr)`
     */
    template <typename Func, typename... Args>
    auto with_write(Func&& fn, Args&&... args) -> decltype(fn(host_ptr(), size(), std::forward<Args>(args)...)) {
        sync_start(/*is_write=*/true);
        struct ScopedGuard {
            DmaBuffer* buf;
            ~ScopedGuard() { if (buf) { try { buf->sync_end(true); } catch (...) {} } }
        } guard{this};
        return fn(host_ptr(), size(), std::forward<Args>(args)...);
    }

    /**
     * @brief Executes a callable (e.g. lambda) with automatic CPU read synchronization.
     * 
     * Automatically calls `sync_start(false)` before invocation and guarantees `sync_end(false)`
     * on scope exit via RAII.
     * 
     * @tparam Func Callable accepting `(const uint8_t* host_ptr, size_t size, Args...)`
     */
    template <typename Func, typename... Args>
    auto with_read(Func&& fn, Args&&... args) const -> decltype(fn(static_cast<const uint8_t*>(host_ptr()), size(), std::forward<Args>(args)...)) {
        const_cast<DmaBuffer*>(this)->sync_start(/*is_write=*/false);
        struct ScopedGuard {
            const DmaBuffer* buf;
            ~ScopedGuard() { if (buf) { try { const_cast<DmaBuffer*>(buf)->sync_end(false); } catch (...) {} } }
        } guard{this};
        return fn(static_cast<const uint8_t*>(host_ptr()), size(), std::forward<Args>(args)...);
    }

    ImageDesc as_image_desc() const {
        ScalixImageDesc c_desc{};
        if (handle_) {
            scalix_dma_buffer_get_desc(handle_, &c_desc);
        }
        return ImageDesc{
            .width = c_desc.width,
            .height = c_desc.height,
            .stride_bytes = c_desc.stride_bytes,
            .format = static_cast<PixelFormat>(c_desc.format),
            .host_ptr = c_desc.host_ptr,
            .data_len = c_desc.data_len,
            .dma_buf_fd = c_desc.dma_buf_fd,
        };
    }

private:
    ScalixDmaBuffer* handle_{nullptr};
};

/**
 * @brief RAII scoped guard for manual DMA buffer CPU cache synchronization.
 */
class ScopedDmaSync {
public:
    explicit ScopedDmaSync(DmaBuffer& buf, bool is_write = true) : buf_(&buf), is_write_(is_write) {
        buf_->sync_start(is_write_);
    }

    ~ScopedDmaSync() {
        if (buf_) {
            try {
                buf_->sync_end(is_write_);
            } catch (...) {}
        }
    }

    ScopedDmaSync(const ScopedDmaSync&) = delete;
    ScopedDmaSync& operator=(const ScopedDmaSync&) = delete;

    ScopedDmaSync(ScopedDmaSync&& other) noexcept : buf_(other.buf_), is_write_(other.is_write_) {
        other.buf_ = nullptr;
    }

private:
    DmaBuffer* buf_{nullptr};
    bool is_write_{true};
};


/**
 * @brief RAII handle for asynchronous tasks returned by Engine::resize_async.
 */
class Task {
public:
    explicit Task(ScalixTask* task) : task_(task) {}
    ~Task() {
        if (task_) {
            scalix_task_release(task_);
            task_ = nullptr;
        }
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task(Task&& other) noexcept : task_(other.task_) {
        other.task_ = nullptr;
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (task_) {
                scalix_task_release(task_);
            }
            task_ = other.task_;
            other.task_ = nullptr;
        }
        return *this;
    }

    /// Checks whether the asynchronous task has finished processing.
    bool is_ready() const {
        return task_ ? scalix_task_is_ready(task_) : false;
    }

    /// Waits for the asynchronous task to complete (timeout_ms=0 for infinite wait).
    void wait(uint32_t timeout_ms = 0, uint8_t* out_dst_ptr = nullptr, size_t out_dst_len = 0) {
        if (!task_) {
            throw std::runtime_error("Task handle is null or already consumed");
        }
        ScalixTask* t = task_;
        task_ = nullptr; // scalix_task_wait consumes the task
        int status = scalix_task_wait(t, timeout_ms, out_dst_ptr, out_dst_len);
        if (status != SCALIX_SUCCESS) {
            throw std::runtime_error("Scalix task wait failed with status code: " + std::to_string(status));
        }
    }

private:
    ScalixTask* task_{nullptr};
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

    /// Enables or disables zero-overhead profiling in the engine.
    void set_profiling(bool enabled) {
        if (engine_) {
            scalix_engine_set_profiling(engine_, enabled);
        }
    }

    /// Retrieves the most recent execution profile if profiling was enabled.
    std::optional<ProfileMetrics> last_profile() const {
        if (!engine_) return std::nullopt;
        ProfileMetrics metrics{};
        int status = scalix_engine_get_last_profile(engine_, &metrics);
        if (status == SCALIX_SUCCESS) {
            return metrics;
        }
        return std::nullopt;
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

    /// Asynchronous execution returning a Task handle
    Task resize_async(const ImageDesc& src, const ImageDesc& dst, Filter filter = Filter::Passthrough) {
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();
        ScalixTask* task = scalix_resize_async(
            engine_,
            &c_src,
            &c_dst,
            static_cast<ScalixFilterMode>(filter)
        );
        if (!task) {
            throw std::runtime_error("Failed to spawn async task in Scalix Engine");
        }
        return Task(task);
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
