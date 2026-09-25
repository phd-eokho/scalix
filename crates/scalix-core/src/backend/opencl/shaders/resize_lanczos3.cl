inline float sinc(float x) {
    if (fabs(x) < 1e-5f) return 1.0f;
    float pi_x = 3.14159265358979323846f * x;
    return sin(pi_x) / pi_x;
}

inline float lanczos3_weight(float x) {
    float ax = fabs(x);
    if (ax >= 3.0f) return 0.0f;
    return sinc(x) * sinc(x / 3.0f);
}

__kernel void resize_lanczos3(
    __global const uchar* src,
    __global uchar* dst,
    uint src_w,
    uint src_h,
    uint dst_w,
    uint dst_h,
    uint src_stride,
    uint dst_stride,
    uint channels
) {
    uint gx = get_global_id(0);
    uint gy = get_global_id(1);
    if (gx >= dst_w || gy >= dst_h) {
        return;
    }

    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    float u = ((float)gx + 0.5f) * scale_x - 0.5f;
    float v = ((float)gy + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floor(u);
    int y0 = (int)floor(v);

    float fx = u - (float)x0;
    float fy = v - (float)y0;

    float wx[6];
    float sum_wx = 0.0f;
    for (int i = 0; i < 6; ++i) {
        wx[i] = lanczos3_weight((float)(i - 2) - fx);
        sum_wx += wx[i];
    }
    float inv_wx = (sum_wx != 0.0f) ? (1.0f / sum_wx) : 1.0f;
    for (int i = 0; i < 6; ++i) {
        wx[i] *= inv_wx;
    }

    float wy[6];
    float sum_wy = 0.0f;
    for (int j = 0; j < 6; ++j) {
        wy[j] = lanczos3_weight((float)(j - 2) - fy);
        sum_wy += wy[j];
    }
    float inv_wy = (sum_wy != 0.0f) ? (1.0f / sum_wy) : 1.0f;
    for (int j = 0; j < 6; ++j) {
        wy[j] *= inv_wy;
    }

    int dst_offset = gy * dst_stride + gx * channels;

    for (uint c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int j = 0; j < 6; ++j) {
            int cy = clamp(y0 - 2 + j, 0, (int)src_h - 1);
            int row_off = cy * src_stride;
            float row_sum = 0.0f;
            for (int i = 0; i < 6; ++i) {
                int cx = clamp(x0 - 2 + i, 0, (int)src_w - 1);
                row_sum += wx[i] * (float)src[row_off + cx * channels + c];
            }
            sum += wy[j] * row_sum;
        }
        dst[dst_offset + c] = (uchar)clamp(round(sum), 0.0f, 255.0f);
    }
}
