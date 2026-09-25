__kernel void resize_nearest(
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

    int sx = clamp((int)round(u), 0, (int)src_w - 1);
    int sy = clamp((int)round(v), 0, (int)src_h - 1);

    int src_offset = sy * src_stride + sx * channels;
    int dst_offset = gy * dst_stride + gx * channels;

    for (uint c = 0; c < channels; ++c) {
        dst[dst_offset + c] = src[src_offset + c];
    }
}
