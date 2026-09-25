__kernel void resize_bilinear(
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

    float fu = floor(u);
    float fv = floor(v);

    int x0 = (int)fu;
    int y0 = (int)fv;

    float fx = u - fu;
    float fy = v - fv;

    int x0_c = clamp(x0, 0, (int)src_w - 1);
    int x1_c = clamp(x0 + 1, 0, (int)src_w - 1);
    int y0_c = clamp(y0, 0, (int)src_h - 1);
    int y1_c = clamp(y0 + 1, 0, (int)src_h - 1);

    int off00 = y0_c * src_stride + x0_c * channels;
    int off10 = y0_c * src_stride + x1_c * channels;
    int off01 = y1_c * src_stride + x0_c * channels;
    int off11 = y1_c * src_stride + x1_c * channels;

    int dst_offset = gy * dst_stride + gx * channels;

    for (uint c = 0; c < channels; ++c) {
        float c00 = (float)src[off00 + c];
        float c10 = (float)src[off10 + c];
        float c01 = (float)src[off01 + c];
        float c11 = (float)src[off11 + c];

        float top = c00 + fx * (c10 - c00);
        float bot = c01 + fx * (c11 - c01);
        float val = top + fy * (bot - top);

        dst[dst_offset + c] = (uchar)clamp(round(val), 0.0f, 255.0f);
    }
}
