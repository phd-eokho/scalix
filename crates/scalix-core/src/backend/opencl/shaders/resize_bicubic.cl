inline void eval_bicubic_weights(float f, float w[4]) {
    float f2 = f * f;
    float f3 = f2 * f;
    w[0] = -0.5f * f3 + f2 - 0.5f * f;
    w[1] = 1.5f * f3 - 2.5f * f2 + 1.0f;
    w[2] = -1.5f * f3 + 2.0f * f2 + 0.5f * f;
    w[3] = 0.5f * f3 - 0.5f * f2;
}

__kernel void resize_bicubic(
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

    float wx[4];
    eval_bicubic_weights(fx, wx);

    float wy[4];
    eval_bicubic_weights(fy, wy);

    int dst_offset = gy * dst_stride + gx * channels;

    for (uint c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int cy = clamp(y0 - 1 + j, 0, (int)src_h - 1);
            int row_off = cy * src_stride;
            float row_sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int cx = clamp(x0 - 1 + i, 0, (int)src_w - 1);
                row_sum += wx[i] * (float)src[row_off + cx * channels + c];
            }
            sum += wy[j] * row_sum;
        }
        dst[dst_offset + c] = (uchar)clamp(round(sum), 0.0f, 255.0f);
    }
}
