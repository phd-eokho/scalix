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
#define SCALIX_ERR_FAILED             -99

/* Opaque Handle Types */
typedef struct ScalixEngine ScalixEngine;
typedef struct ScalixTask ScalixTask;

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

/* Engine Lifecycle */
ScalixEngine* scalix_engine_create(ScalixBackendType backend);
void scalix_engine_destroy(ScalixEngine* engine);

/* 1. Synchronous Execution */
int scalix_resize_sync(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    ScalixImageDesc* dst,
    ScalixFilterMode filter
);

/* 2. Asynchronous Execution (Task Handle / Polling / Wait) */
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
int scalix_resize_submit(
    ScalixEngine* engine,
    const ScalixImageDesc* src,
    const ScalixImageDesc* dst,
    ScalixFilterMode filter,
    ScalixCompletionCallback callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif /* SCALIX_H */
