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
#define SCALIX_ERR_UNALIGNED_POINTER   -12
#define SCALIX_ERR_FAILED             -99

#define SCALIX_REQUIRED_ALIGNMENT_BYTES 64

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
    uint32_t max_mip_levels; /* 0 = automatic / unlimited, >0 = limit mipchain depth */
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

/*
 * Creates a new Scalix Engine instance with default PID thread naming (<pid>/scx-*).
 * 
 * backend: Hardware accelerator backend type to initialize.
 * Returns: ScalixEngine* Pointer to created engine handle, or NULL on failure.
 */
ScalixEngine* scalix_engine_create(ScalixBackendType backend);

/*
 * Creates a new Scalix Engine instance with a custom thread naming prefix.
 * 
 * Internal threads are named <prefix>/scx-hw (hardware executor) and
 * <prefix>/scx-w<id> (callback/worker pool).
 * 
 * backend: Hardware accelerator backend type to initialize.
 * thread_prefix: Custom prefix string for naming internal engine threads.
 *                Pass NULL or empty string to default to process ID (<pid>).
 * Note: Maximum effective length of thread_prefix is 7 characters.
 * Warning: Prefixes exceeding 7 characters will be automatically truncated to 7 characters
 *          with a runtime warning log to strictly adhere to the Linux 15-character thread
 *          name (comm) limit.
 * Returns: ScalixEngine* Pointer to created engine handle, or NULL on failure.
 */
ScalixEngine* scalix_engine_create_with_prefix(
    ScalixBackendType backend,
    const char* thread_prefix
);

/*
 * Destroys a Scalix Engine instance and releases all associated worker threads.
 * 
 * engine: Pointer to engine handle to destroy.
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

/*
 * Enables or disables zero-overhead profiling in the engine.
 * 
 * engine: Pointer to engine handle.
 * enabled: True to enable latency breakdowns and GPU hardware timestamps.
 * Returns: SCALIX_SUCCESS on success, error code otherwise.
 */
int scalix_engine_set_profiling(ScalixEngine* engine, bool enabled);

/*
 * Retrieves the most recent profile metrics if profiling was enabled.
 * 
 * engine: Pointer to engine handle.
 * out_metrics: Pointer to metrics structure to populate.
 * Returns: SCALIX_SUCCESS on success, error code otherwise.
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
/*
 * Checks whether an asynchronous task has completed without blocking.
 * Returns true if completed, false otherwise.
 */
bool scalix_task_is_ready(const ScalixTask* task);

/*
 * Waits for an asynchronous task to complete.
 *
 * task: Task handle returned by scalix_resize_async*.
 * timeout_ms: Timeout in milliseconds (0 or UINT32_MAX = wait indefinitely, >0 = timeout limit).
 * out_dst_ptr: Optional buffer to copy destination image pixels into upon completion (can be NULL).
 * out_dst_len: Length of out_dst_ptr buffer in bytes.
 * Returns: SCALIX_SUCCESS on success, SCALIX_ERR_TIMEOUT on timeout, or negative error code.
 */
int scalix_task_wait(
    ScalixTask* task,
    uint32_t timeout_ms,
    uint8_t* out_dst_ptr,
    size_t out_dst_len
);

/*
 * Releases and deallocates an asynchronous task handle.
 * Must be called to prevent memory leaks after task wait or when abandoning a task.
 */
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
/*
 * Allocates a hardware-backed DMA buffer (Linux DMA-Heap/DRM, Android AHardwareBuffer).
 * 
 * width: Image buffer width in pixels.
 * height: Image buffer height in pixels.
 * format: Pixel format of the buffer.
 * Returns: ScalixDmaBuffer* Pointer to created DMA buffer handle, or NULL on failure.
 */
ScalixDmaBuffer* scalix_dma_buffer_allocate(
    uint32_t width,
    uint32_t height,
    ScalixPixelFormat format
);

/*
 * Releases a DMA buffer and unmaps its virtual memory.
 * 
 * buffer: Pointer to DMA buffer handle to release.
 */
void scalix_dma_buffer_free(ScalixDmaBuffer* buffer);

/* Returns the underlying Linux DMA-BUF file descriptor (or -1 if not applicable). */
int scalix_dma_buffer_get_fd(const ScalixDmaBuffer* buffer);

/* Returns the memory-mapped CPU virtual address pointer. */
uint8_t* scalix_dma_buffer_get_host_ptr(const ScalixDmaBuffer* buffer);

/* Returns the total allocated byte size of the DMA buffer. */
size_t scalix_dma_buffer_get_size(const ScalixDmaBuffer* buffer);

/* Returns the row pitch/stride in bytes. */
size_t scalix_dma_buffer_get_stride(const ScalixDmaBuffer* buffer);

/* Populates a ScalixImageDesc referencing this DMA buffer. */
int scalix_dma_buffer_get_desc(
    const ScalixDmaBuffer* buffer,
    ScalixImageDesc* out_desc
);

/*
 * Prepares DMA buffer for CPU read/write access (cache invalidation/clean).
 * 
 * buffer: DMA buffer handle.
 * is_write: True if CPU will write to buffer, False for read-only.
 */
int scalix_dma_buffer_sync_start(const ScalixDmaBuffer* buffer, bool is_write);

/*
 * Concludes CPU read/write access to flush caches for hardware accelerators.
 * 
 * buffer: DMA buffer handle.
 * is_write: True if CPU wrote to buffer, False for read-only.
 */
int scalix_dma_buffer_sync_end(const ScalixDmaBuffer* buffer, bool is_write);

#ifdef __cplusplus
}
#endif

#endif /* SCALIX_H */
