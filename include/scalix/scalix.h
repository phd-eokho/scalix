#ifndef SCALIX_H
#define SCALIX_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return Status Codes */
#define SCALIX_SUCCESS                  0
#define SCALIX_ERR_NULL_PTR            -1
#define SCALIX_ERR_INVALID_DIMENSIONS  -2
#define SCALIX_ERR_INVALID_STRIDE      -3
#define SCALIX_ERR_BUFFER_TOO_SMALL    -4
#define SCALIX_ERR_UNSUPPORTED_FORMAT  -5
#define SCALIX_ERR_BACKEND_UNAVAILABLE -6
#define SCALIX_ERR_TIMEOUT             -7
#define SCALIX_ERR_DMA_UNAVAILABLE     -8
#define SCALIX_ERR_DMA_ALLOCATION_FAILED -9
#define SCALIX_ERR_DMA_MAP_FAILED      -10
#define SCALIX_ERR_DMA_SYNC_FAILED     -11
#define SCALIX_ERR_FAILED             -99

/* Opaque Handle Types */
typedef struct ScalixEngine ScalixEngine;
typedef struct ScalixTask ScalixTask;
typedef struct ScalixDmaBuffer ScalixDmaBuffer;

/* Backend Provider Types */
typedef enum ScalixBackendType {
    SCALIX_BACKEND_AUTO        = 0,
    SCALIX_BACKEND_VULKAN      = 1,
    SCALIX_BACKEND_OPENGL      = 2,
    SCALIX_BACKEND_NPU         = 3,
    SCALIX_BACKEND_HW2D        = 4,
    SCALIX_BACKEND_CPU         = 5,
    SCALIX_BACKEND_PASSTHROUGH = 6
} ScalixBackendType;

/* Execution Pipeline Strategy */
typedef enum ScalixStrategy {
    SCALIX_STRATEGY_AUTO        = 0,
    SCALIX_STRATEGY_BLIT        = 1,
    SCALIX_STRATEGY_RASTER      = 2,
    SCALIX_STRATEGY_LOD_PYRAMID = 3,
    SCALIX_STRATEGY_COMPUTE     = 4
} ScalixStrategy;

/* Interpolation / Scaling Filters */
typedef enum ScalixFilterMode {
    SCALIX_FILTER_NEAREST     = 0,
    SCALIX_FILTER_BILINEAR    = 1,
    SCALIX_FILTER_BICUBIC     = 2,
    SCALIX_FILTER_LANCZOS3    = 3,
    SCALIX_FILTER_AREA        = 4,
    SCALIX_FILTER_PASSTHROUGH = 100
} ScalixFilterMode;

/* Pixel Formats */
typedef enum ScalixPixelFormat {
    SCALIX_FORMAT_RGBA8888 = 0,
    SCALIX_FORMAT_BGRA8888 = 1,
    SCALIX_FORMAT_RGB888   = 2,
    SCALIX_FORMAT_BGR888   = 3,
    SCALIX_FORMAT_R8       = 4,
    SCALIX_FORMAT_RG88     = 5,
    SCALIX_FORMAT_NV12     = 6,
    SCALIX_FORMAT_YUV420P  = 7,
    SCALIX_FORMAT_RGBA16F  = 8,
    SCALIX_FORMAT_RGBA32F  = 9
} ScalixPixelFormat;

/* Vulkan Backend Options */
typedef struct ScalixVulkanOptions {
    ScalixStrategy strategy;
} ScalixVulkanOptions;

/* Dynamic Resize Options */
typedef struct ScalixResizeOptions {
    ScalixFilterMode filter;
    ScalixVulkanOptions vulkan;
} ScalixResizeOptions;

/* Image Descriptor */
typedef struct ScalixImageDesc {
    uint32_t width;
    uint32_t height;
    size_t stride_bytes;
    ScalixPixelFormat format;
    uint8_t* host_ptr;
    size_t data_len;
    int dma_buf_fd;
} ScalixImageDesc;

/* Completion Callback Signature */
typedef void (*ScalixCompletionCallback)(int status_code, void* user_data);

/**
 * @brief Creates a new Scalix Engine instance with default PID thread naming (`<pid>/scx-*`).
 * 
 * @param backend Hardware accelerator backend type to initialize.
 * @return ScalixEngine* Pointer to created engine handle, or NULL on failure.
 */
ScalixEngine* scalix_engine_create(ScalixBackendType backend);

/**
 * @brief Creates a new Scalix Engine instance with a custom thread naming prefix.
 * 
 * Internal threads are named `<prefix>/scx-hw` (hardware executor) and
 * `<prefix>/scx-w<id>` (callback/worker pool).
 * 
 * @param backend Hardware accelerator backend type to initialize.
 * @param thread_prefix Custom prefix string for naming internal engine threads.
 *                      Pass NULL or empty string to default to process ID (`<pid>`).
 * @note Maximum effective length of @p thread_prefix is 7 characters.
 * @warning Prefixes exceeding 7 characters will be automatically truncated to 7 characters
 *          with a runtime warning log to strictly adhere to the Linux 15-character thread
 *          name (`comm`) limit.
 * @return ScalixEngine* Pointer to created engine handle, or NULL on failure.
 */
ScalixEngine* scalix_engine_create_with_prefix(
    ScalixBackendType backend,
    const char* thread_prefix
);

/**
 * @brief Destroys a Scalix Engine instance and releases all associated worker threads.
 * 
 * @param engine Pointer to engine handle to destroy.
 */
void scalix_engine_destroy(ScalixEngine* engine);

/* Profiling Metrics */
typedef struct ScalixProfileMetrics {
    double host_unpack_ms;
    double gpu_upload_ms;
    double gpu_pure_blit_ms;
    double gpu_download_ms;
    double host_repack_ms;
    double driver_sync_ms;
    double total_wall_ms;
} ScalixProfileMetrics;

/**
 * @brief Enables or disables zero-overhead profiling in the engine.
 * 
 * @param engine Pointer to engine handle.
 * @param enabled True to enable latency breakdowns and GPU hardware timestamps.
 * @return SCALIX_SUCCESS on success, error code otherwise.
 */
int scalix_engine_set_profiling(ScalixEngine* engine, bool enabled);

/**
 * @brief Retrieves the most recent profile metrics if profiling was enabled.
 * 
 * @param engine Pointer to engine handle.
 * @param out_metrics Pointer to metrics structure to populate.
 * @return SCALIX_SUCCESS on success, error code otherwise.
 */
int scalix_engine_get_last_profile(
    const ScalixEngine* engine,
    ScalixProfileMetrics* out_metrics
);

/* 1. Synchronous Execution */
int scalix_resize_sync_with_options(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    ScalixImageDesc* dst,
    const ScalixResizeOptions* options
);

int scalix_resize_sync(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    ScalixImageDesc* dst,
    ScalixFilterMode filter
);

/* 2. Asynchronous Execution (Task Handle / Polling / Wait) */
ScalixTask* scalix_resize_async_with_options(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    const ScalixImageDesc* dst,
    const ScalixResizeOptions* options
);

ScalixTask* scalix_resize_async(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    const ScalixImageDesc* dst,
    ScalixFilterMode filter
);
bool scalix_task_is_ready(const ScalixTask* task);
int scalix_task_wait(
    ScalixTask* task,
    uint32_t timeout_ms,
    uint8_t* out_dst_ptr,
    size_t out_dst_len
);
void scalix_task_release(ScalixTask* task);

/* 3. Callback-driven Execution */
int scalix_resize_submit_with_options(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    const ScalixImageDesc* dst,
    const ScalixResizeOptions* options,
    ScalixCompletionCallback callback,
    void* user_data
);

int scalix_resize_submit(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    const ScalixImageDesc* dst,
    ScalixFilterMode filter,
    ScalixCompletionCallback callback,
    void* user_data
);

/* 4. Zero-Copy DMA Buffer Lifecycle & Synchronization */
/**
 * @brief Allocates a hardware-backed DMA buffer (Linux DMA-Heap/DRM, Android AHardwareBuffer).
 * 
 * @param width Image buffer width in pixels.
 * @param height Image buffer height in pixels.
 * @param format Pixel format of the buffer.
 * @return ScalixDmaBuffer* Pointer to created DMA buffer handle, or NULL on failure.
 */
ScalixDmaBuffer* scalix_dma_buffer_allocate(
    uint32_t width,
    uint32_t height,
    ScalixPixelFormat format
);

/**
 * @brief Releases a DMA buffer and unmaps its virtual memory.
 * 
 * @param buffer Pointer to DMA buffer handle to release.
 */
void scalix_dma_buffer_free(ScalixDmaBuffer* buffer);

/**
 * @brief Returns the underlying Linux DMA-BUF file descriptor (or -1 if not applicable).
 */
int scalix_dma_buffer_get_fd(const ScalixDmaBuffer* buffer);

/**
 * @brief Returns the memory-mapped CPU virtual address pointer.
 */
uint8_t* scalix_dma_buffer_get_host_ptr(const ScalixDmaBuffer* buffer);

/**
 * @brief Returns the total allocated byte size of the DMA buffer.
 */
size_t scalix_dma_buffer_get_size(const ScalixDmaBuffer* buffer);

/**
 * @brief Returns the row pitch/stride in bytes.
 */
size_t scalix_dma_buffer_get_stride(const ScalixDmaBuffer* buffer);

/**
 * @brief Populates a ScalixImageDesc referencing this DMA buffer.
 */
int scalix_dma_buffer_get_desc(
    const ScalixDmaBuffer* buffer,
    ScalixImageDesc* out_desc
);

/**
 * @brief Prepares DMA buffer for CPU read/write access (cache invalidation/clean).
 * 
 * @param buffer DMA buffer handle.
 * @param is_write True if CPU will write to buffer, False for read-only.
 */
int scalix_dma_buffer_sync_start(const ScalixDmaBuffer* buffer, bool is_write);

/**
 * @brief Concludes CPU read/write access to flush caches for hardware accelerators.
 * 
 * @param buffer DMA buffer handle.
 * @param is_write True if CPU wrote to buffer, False for read-only.
 */
int scalix_dma_buffer_sync_end(const ScalixDmaBuffer* buffer, bool is_write);

#ifdef __cplusplus
}
#endif

#endif /* SCALIX_H */
