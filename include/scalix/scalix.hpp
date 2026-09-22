#pragma once

#include <scalix/scalix.h>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

namespace scalix {

/// Custom allocator ensuring 64-byte alignment for cache lines, SIMD, and GPU zero-copy DMA compatibility.
template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    template <typename U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template <typename U> bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, SCALIX_REQUIRED_ALIGNMENT_BYTES, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }
};

/// Type alias for standard vector using 64-byte aligned allocator.
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

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

enum class Strategy : int {
    Auto        = SCALIX_STRATEGY_AUTO,
    Blit        = SCALIX_STRATEGY_BLIT,
    Raster      = SCALIX_STRATEGY_RASTER,
    LodPyramid  = SCALIX_STRATEGY_LOD_PYRAMID,
    Compute     = SCALIX_STRATEGY_COMPUTE
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

struct ImageDesc final {
    uint32_t width{0};
    uint32_t height{0};
    size_t stride_bytes{0};
    PixelFormat format{PixelFormat::Rgba8888};
    uint8_t* host_ptr{nullptr};
    size_t data_len{0};
    int dma_buf_fd{-1};

    [[nodiscard]] ScalixImageDesc to_c() const {
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

/// @brief RAII wrapper for a hardware-backed zero-copy DMA buffer.
class DmaBuffer final {
public:
    /// @brief Allocates a new hardware DMA buffer.
    /// @param width Width in pixels.
    /// @param height Height in pixels.
    /// @param format Pixel format (default: PixelFormat::Rgba8888).
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

    [[nodiscard]] int fd() const noexcept {
        return handle_ ? scalix_dma_buffer_get_fd(handle_) : -1;
    }

    [[nodiscard]] uint8_t* host_ptr() const noexcept {
        return handle_ ? scalix_dma_buffer_get_host_ptr(handle_) : nullptr;
    }

    [[nodiscard]] size_t size() const noexcept {
        return handle_ ? scalix_dma_buffer_get_size(handle_) : 0;
    }

    [[nodiscard]] size_t stride() const noexcept {
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

    /// @brief Executes a callable (e.g. lambda) with automatic CPU write synchronization.
    /// @tparam Func Callable accepting `(uint8_t* host_ptr, size_t size, Args...)` or `(uint8_t* host_ptr)`.
    /// @param fn Callable to invoke inside synchronized memory scope.
    /// @param args Forwarded arguments to the callable.
    /// @return The result of calling `fn`.
    /// @note Automatically calls `sync_start(true)` before invocation and guarantees `sync_end(true)`
    ///       on scope exit via RAII (even if an exception occurs).
    template <typename Func, typename... Args>
    auto with_write(Func&& fn, Args&&... args) -> decltype(fn(host_ptr(), size(), std::forward<Args>(args)...)) {
        sync_start(/*is_write=*/true);
        struct ScopedGuard {
            DmaBuffer* buf;
            ~ScopedGuard() { if (buf) { try { buf->sync_end(true); } catch (...) {} } }
        } guard{this};
        return fn(host_ptr(), size(), std::forward<Args>(args)...);
    }

    /// @brief Executes a callable (e.g. lambda) with automatic CPU read synchronization.
    /// @tparam Func Callable accepting `(const uint8_t* host_ptr, size_t size, Args...)`.
    /// @param fn Callable to invoke inside synchronized memory scope.
    /// @param args Forwarded arguments to the callable.
    /// @return The result of calling `fn`.
    /// @note Automatically calls `sync_start(false)` before invocation and guarantees `sync_end(false)`
    ///       on scope exit via RAII.
    template <typename Func, typename... Args>
    auto with_read(Func&& fn, Args&&... args) const -> decltype(fn(static_cast<const uint8_t*>(host_ptr()), size(), std::forward<Args>(args)...)) {
        const_cast<DmaBuffer*>(this)->sync_start(/*is_write=*/false);
        struct ScopedGuard {
            const DmaBuffer* buf;
            ~ScopedGuard() { if (buf) { try { const_cast<DmaBuffer*>(buf)->sync_end(false); } catch (...) {} } }
        } guard{this};
        return fn(static_cast<const uint8_t*>(host_ptr()), size(), std::forward<Args>(args)...);
    }

    [[nodiscard]] ImageDesc as_image_desc() const {
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

/// RAII scoped guard for manual DMA buffer CPU cache synchronization.
class ScopedDmaSync final {
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

/// RAII handle for asynchronous tasks returned by Engine::resize_async.
class Task final {
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

    /// @brief Checks whether the asynchronous task has finished processing.
    /// @return True if task has completed, false otherwise.
    [[nodiscard]] bool is_ready() const {
        return task_ ? scalix_task_is_ready(task_) : false;
    }

    /// @brief Waits for the asynchronous task to complete.
    /// @param timeout_ms Timeout duration in milliseconds (0 = non-blocking poll, UINT64_MAX = infinite).
    /// @param out_dst_ptr Optional pointer to copy destination image data into upon completion.
    /// @param out_dst_len Length of out_dst_ptr buffer in bytes.
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

struct VulkanOptions final {
    Strategy strategy{Strategy::Auto};
    uint32_t max_mip_levels{0}; // 0 = automatic / unlimited, >0 = limit mipchain depth

    [[nodiscard]] ScalixVulkanOptions to_c() const {
        return ScalixVulkanOptions{
            .strategy = static_cast<ScalixStrategy>(strategy),
            .max_mip_levels = max_mip_levels,
        };
    }
};

struct ResizeOptions final {
    Filter filter{Filter::Passthrough};
    VulkanOptions vulkan{};

    [[nodiscard]] ScalixResizeOptions to_c() const {
        return ScalixResizeOptions{
            .filter = static_cast<ScalixFilterMode>(filter),
            .vulkan = vulkan.to_c(),
        };
    }
};

class Engine final {
public:
    /// @brief Constructs a new Scalix Engine instance.
    /// @param backend Hardware accelerator backend type (default: Backend::Auto).
    /// @param thread_prefix Optional custom naming prefix for engine worker threads
    ///                      (e.g., `<prefix>/scx-hw` and `<prefix>/scx-w<id>`). If nullptr
    ///                      or empty, defaults to current process ID (`<pid>`).
    /// @note Maximum effective length of @p thread_prefix is 7 characters.
    /// @warning Prefixes longer than 7 characters are automatically truncated to 7 characters
    ///          to guarantee strict compliance with Linux OS 15-character thread name (`comm`) limit.
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

    /// @brief Enables or disables zero-overhead profiling in the engine.
    /// @param enabled True to enable execution profiling, false to disable.
    void set_profiling(bool enabled) {
        if (engine_) {
            scalix_engine_set_profiling(engine_, enabled);
        }
    }

    /// @brief Retrieves the most recent execution profile if profiling was enabled.
    /// @return Optional ProfileMetrics containing stage-by-stage timings.
    [[nodiscard]] std::optional<ProfileMetrics> last_profile() const {
        if (!engine_) return std::nullopt;
        ProfileMetrics metrics{};
        int status = scalix_engine_get_last_profile(engine_, &metrics);
        if (status == SCALIX_SUCCESS) {
            return metrics;
        }
        return std::nullopt;
    }

    /// @brief Synchronously resizes an image with explicit options.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param options Resize options (filter, strategy, mipmap levels).
    void resize(const ImageDesc& src, ImageDesc& dst, const ResizeOptions& options) {
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();
        auto c_opt = options.to_c();
        int status = scalix_resize_sync_with_options(
            engine_,
            &c_src,
            &c_dst,
            &c_opt
        );
        if (status != SCALIX_SUCCESS) {
            throw std::runtime_error("Scalix resize_sync failed with status code: " + std::to_string(status));
        }
    }

    /// @brief Synchronously resizes an image with a specific filter.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode (default: Filter::Passthrough).
    void resize(const ImageDesc& src, ImageDesc& dst, Filter filter = Filter::Passthrough) {
        resize(src, dst, ResizeOptions{.filter = filter});
    }

    /// @brief Synchronously resizes an image with filter and pipeline strategy.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode.
    /// @param strategy Vulkan execution strategy.
    void resize(const ImageDesc& src, ImageDesc& dst, Filter filter, Strategy strategy) {
        resize(src, dst, ResizeOptions{.filter = filter, .vulkan = {.strategy = strategy}});
    }

    /// @brief Spawns an asynchronous image resize task returning a Task handle.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param options Resize options (filter, strategy, mipmap levels).
    /// @return RAII Task handle for polling or awaiting completion.
    [[nodiscard]] Task resize_async(const ImageDesc& src, const ImageDesc& dst, const ResizeOptions& options) {
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();
        auto c_opt = options.to_c();
        ScalixTask* task = scalix_resize_async_with_options(
            engine_,
            &c_src,
            &c_dst,
            &c_opt
        );
        if (!task) {
            throw std::runtime_error("Failed to spawn async task in Scalix Engine");
        }
        return Task(task);
    }

    /// @brief Spawns an asynchronous image resize task with a specific filter.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode (default: Filter::Passthrough).
    /// @return RAII Task handle for polling or awaiting completion.
    [[nodiscard]] Task resize_async(const ImageDesc& src, const ImageDesc& dst, Filter filter = Filter::Passthrough) {
        return resize_async(src, dst, ResizeOptions{.filter = filter});
    }

    /// @brief Spawns an asynchronous image resize task with filter and pipeline strategy.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode.
    /// @param strategy Vulkan execution strategy.
    /// @return RAII Task handle for polling or awaiting completion.
    [[nodiscard]] Task resize_async(const ImageDesc& src, const ImageDesc& dst, Filter filter, Strategy strategy) {
        return resize_async(src, dst, ResizeOptions{.filter = filter, .vulkan = {.strategy = strategy}});
    }

    /// @brief Submits a callback-driven asynchronous image resize task.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param options Resize options (filter, strategy, mipmap levels).
    /// @param callback Invoked upon task completion with status code (0 = success).
    void resize_callback(
        const ImageDesc& src,
        const ImageDesc& dst,
        const ResizeOptions& options,
        std::function<void(int status)> callback
    ) {
        auto* cb_ptr = new std::function<void(int status)>(std::move(callback));
        auto c_src = src.to_c();
        auto c_dst = dst.to_c();
        auto c_opt = options.to_c();

        int status = scalix_resize_submit_with_options(
            engine_,
            &c_src,
            &c_dst,
            &c_opt,
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

    /// @brief Submits a callback-driven asynchronous image resize task with filter.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode.
    /// @param callback Invoked upon task completion with status code.
    void resize_callback(
        const ImageDesc& src,
        const ImageDesc& dst,
        Filter filter,
        std::function<void(int status)> callback
    ) {
        resize_callback(src, dst, ResizeOptions{.filter = filter}, std::move(callback));
    }

    /// @brief Submits a callback-driven asynchronous image resize task with filter and strategy.
    /// @param src Source image descriptor.
    /// @param dst Destination image descriptor.
    /// @param filter Scaling filter mode.
    /// @param strategy Vulkan execution strategy.
    /// @param callback Invoked upon task completion with status code.
    void resize_callback(
        const ImageDesc& src,
        const ImageDesc& dst,
        Filter filter,
        Strategy strategy,
        std::function<void(int status)> callback
    ) {
        resize_callback(
            src,
            dst,
            ResizeOptions{.filter = filter, .vulkan = {.strategy = strategy}},
            std::move(callback)
        );
    }

private:
    ScalixEngine* engine_{nullptr};
};

} // namespace scalix
