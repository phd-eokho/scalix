__kernel void resize_area(
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

    float x0 = (float)gx * scale_x;
    float x1 = (float)(gx + 1) * scale_x;
    float y0 = (float)gy * scale_y;
    float y1 = (float)(gy + 1) * scale_y;

    int sx_min = (int)floor(x0);
    int sx_max = (int)floor(x1);
    if ((float)sx_max == x1 && sx_max > sx_min) {
        sx_max--;
    }

    int sy_min = (int)floor(y0);
    int sy_max = (int)floor(y1);
    if ((float)sy_max == y1 && sy_max > sy_min) {
        sy_max--;
    }

    float sum_col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float total_weight = 0.0f;

    for (int sy = sy_min; sy <= sy_max; ++sy) {
        float wy = max(0.0f, min((float)sy + 1.0f, y1) - max((float)sy, y0));
        int cy = clamp(sy, 0, (int)src_h - 1);
        int row_off = cy * src_stride;

        for (int sx = sx_min; sx <= sx_max; ++sx) {
            float wx = max(0.0f, min((float)sx + 1.0f, x1) - max((float)sx, x0));
            float w = wx * wy;
            int cx = clamp(sx, 0, (int)src_w - 1);
            int src_off = row_off + cx * channels;

            for (uint c = 0; c < channels && c < 4; ++c) {
                sum_col[c] += (float)src[src_off + c] * w;
            }
            total_weight += w;
        }
    }

    int dst_offset = gy * dst_stride + gx * channels;
    float inv_w = (total_weight > 0.0f) ? (1.0f / total_weight) : 1.0f;

    for (uint c = 0; c < channels && c < 4; ++c) {
        float val = sum_col[c] * inv_w;
        dst[dst_offset + c] = (uchar)clamp(round(val), 0.0f, 255.0f);
    }
}
