//! GPU Compute Shader Unpacker / Repacker for Packed 24-bit RGB (RGB888) Formats
//!
//! Offloads 3-byte RGB <-> 4-byte RGBA unpacking and repacking from host CPU to GPU compute shaders (`vkCmdDispatch`).
//! Avoids expensive host byte manipulation loops and reduces PCIe staging transfer bandwidth by 25%.

use crate::backend::vulkan::context::VulkanContext;
use crate::types::{Result, ScalixError};
use ash::vk;
use std::sync::Arc;

/// Pre-compiled SPIR-V binary bytecode for RGB888 -> RGBA8888 compute unpack kernel.
///
/// ```glsl
/// #version 450
/// layout(local_size_x = 16, local_size_y = 16) in;
/// layout(set = 0, binding = 0) readonly buffer SrcBuffer { uint src_data[]; };
/// layout(set = 0, binding = 1, rgba8) uniform writeonly image2D dst_image;
/// layout(push_constant) uniform PushConsts { uint width; uint height; };
///
/// void main() {
///     uint x = gl_GlobalInvocationID.x;
///     uint y = gl_GlobalInvocationID.y;
///     if (x >= width || y >= height) return;
///     uint pixel_idx = y * width + x;
///     uint byte_offset = pixel_idx * 3;
///     uint word_idx = byte_offset >> 2;
///     uint w0 = src_data[word_idx];
///     uint w1 = src_data[word_idx + 1];
///     uint r, g, b;
///     uint b_mod = byte_offset & 3;
///     if (b_mod == 0) {
///         r = w0 & 0xFFu;
///         g = (w0 >> 8) & 0xFFu;
///         b = (w0 >> 16) & 0xFFu;
///     } else if (b_mod == 1) {
///         r = (w0 >> 8) & 0xFFu;
///         g = (w0 >> 16) & 0xFFu;
///         b = (w0 >> 24) & 0xFFu;
///     } else if (b_mod == 2) {
///         r = (w0 >> 16) & 0xFFu;
///         g = (w0 >> 24) & 0xFFu;
///         b = w1 & 0xFFu;
///     } else {
///         r = (w0 >> 24) & 0xFFu;
///         g = w1 & 0xFFu;
///         b = (w1 >> 8) & 0xFFu;
///     }
///     vec4 color = vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
///     imageStore(dst_image, ivec2(x, y), color);
/// }
/// ```
pub const UNPACK_RGB888_COMP_SPV: &[u32] = &[
    119734787, 65536, 524299, 165, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 16, 16, 1, 196611, 2, 450,
    262149, 4, 1852399981, 0, 196613, 8, 120, 524293, 11, 1197436007, 1633841004, 1986939244,
    1952539503, 1231974249, 68, 196613, 16, 121, 327685, 22, 1752397136, 1936617283, 29556, 327686,
    22, 0, 1952737655, 104, 327686, 22, 1, 1734960488, 29800, 196613, 24, 0, 327685, 43,
    1702390128, 1684627308, 120, 327685, 50, 1702132066, 1717989215, 7628147, 327685, 54,
    1685221239, 2019846495, 0, 262149, 58, 1718184051, 116, 196613, 63, 12407, 327685, 65,
    1113813587, 1701209717, 114, 393222, 65, 0, 1600352883, 1635017060, 0, 196613, 67, 0, 196613,
    72, 12663, 196613, 82, 114, 196613, 86, 103, 196613, 91, 98, 262149, 139, 1869377379, 114,
    327685, 154, 1601467236, 1734438249, 101, 262215, 11, 11, 28, 196679, 22, 2, 327752, 22, 0, 35,
    0, 327752, 22, 1, 35, 4, 262215, 64, 6, 4, 196679, 65, 3, 262216, 65, 0, 24, 327752, 65, 0, 35,
    0, 196679, 67, 24, 262215, 67, 33, 0, 262215, 67, 34, 0, 196679, 154, 25, 262215, 154, 33, 1,
    262215, 154, 34, 0, 262215, 164, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7,
    7, 6, 262167, 9, 6, 3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6,
    262187, 6, 17, 1, 131092, 20, 262174, 22, 6, 6, 262176, 23, 9, 22, 262203, 23, 24, 9, 262165,
    25, 32, 1, 262187, 25, 26, 0, 262176, 27, 9, 6, 262187, 25, 35, 1, 262187, 6, 52, 3, 262187,
    25, 56, 2, 262187, 6, 61, 8, 196637, 64, 6, 196638, 65, 64, 262176, 66, 2, 65, 262203, 66, 67,
    2, 262176, 69, 2, 6, 262187, 6, 84, 255, 262187, 25, 88, 8, 262187, 25, 93, 16, 262187, 25,
    109, 24, 262187, 6, 115, 2, 196630, 136, 32, 262167, 137, 136, 4, 262176, 138, 7, 137, 262187,
    136, 142, 1132396544, 262187, 136, 150, 1065353216, 589849, 152, 136, 1, 0, 0, 0, 2, 4, 262176,
    153, 0, 152, 262203, 153, 154, 0, 262167, 160, 25, 2, 262187, 6, 163, 16, 393260, 9, 164, 163,
    163, 17, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 262203, 7, 16, 7, 262203, 7, 43, 7,
    262203, 7, 50, 7, 262203, 7, 54, 7, 262203, 7, 58, 7, 262203, 7, 63, 7, 262203, 7, 72, 7,
    262203, 7, 82, 7, 262203, 7, 86, 7, 262203, 7, 91, 7, 262203, 138, 139, 7, 327745, 13, 14, 11,
    12, 262205, 6, 15, 14, 196670, 8, 15, 327745, 13, 18, 11, 17, 262205, 6, 19, 18, 196670, 16,
    19, 262205, 6, 21, 8, 327745, 27, 28, 24, 26, 262205, 6, 29, 28, 327854, 20, 30, 21, 29,
    262312, 20, 31, 30, 196855, 33, 0, 262394, 31, 32, 33, 131320, 32, 262205, 6, 34, 16, 327745,
    27, 36, 24, 35, 262205, 6, 37, 36, 327854, 20, 38, 34, 37, 131321, 33, 131320, 33, 458997, 20,
    39, 30, 5, 38, 32, 196855, 41, 0, 262394, 39, 40, 41, 131320, 40, 65789, 131320, 41, 262205, 6,
    44, 16, 327745, 27, 45, 24, 26, 262205, 6, 46, 45, 327812, 6, 47, 44, 46, 262205, 6, 48, 8,
    327808, 6, 49, 47, 48, 196670, 43, 49, 262205, 6, 51, 43, 327812, 6, 53, 51, 52, 196670, 50,
    53, 262205, 6, 55, 50, 327874, 6, 57, 55, 56, 196670, 54, 57, 262205, 6, 59, 50, 327879, 6, 60,
    59, 52, 327812, 6, 62, 60, 61, 196670, 58, 62, 262205, 6, 68, 54, 393281, 69, 70, 67, 26, 68,
    262205, 6, 71, 70, 196670, 63, 71, 262205, 6, 73, 54, 327808, 6, 74, 73, 17, 393281, 69, 75,
    67, 26, 74, 262205, 6, 76, 75, 196670, 72, 76, 262205, 6, 77, 50, 327879, 6, 78, 77, 52,
    327850, 20, 79, 78, 12, 196855, 81, 0, 262394, 79, 80, 96, 131320, 80, 262205, 6, 83, 63,
    327879, 6, 85, 83, 84, 196670, 82, 85, 262205, 6, 87, 63, 327874, 6, 89, 87, 88, 327879, 6, 90,
    89, 84, 196670, 86, 90, 262205, 6, 92, 63, 327874, 6, 94, 92, 93, 327879, 6, 95, 94, 84,
    196670, 91, 95, 131321, 81, 131320, 96, 262205, 6, 97, 50, 327879, 6, 98, 97, 52, 327850, 20,
    99, 98, 17, 196855, 101, 0, 262394, 99, 100, 112, 131320, 100, 262205, 6, 102, 63, 327874, 6,
    103, 102, 88, 327879, 6, 104, 103, 84, 196670, 82, 104, 262205, 6, 105, 63, 327874, 6, 106,
    105, 93, 327879, 6, 107, 106, 84, 196670, 86, 107, 262205, 6, 108, 63, 327874, 6, 110, 108,
    109, 327879, 6, 111, 110, 84, 196670, 91, 111, 131321, 101, 131320, 112, 262205, 6, 113, 50,
    327879, 6, 114, 113, 52, 327850, 20, 116, 114, 115, 196855, 118, 0, 262394, 116, 117, 127,
    131320, 117, 262205, 6, 119, 63, 327874, 6, 120, 119, 93, 327879, 6, 121, 120, 84, 196670, 82,
    121, 262205, 6, 122, 63, 327874, 6, 123, 122, 109, 327879, 6, 124, 123, 84, 196670, 86, 124,
    262205, 6, 125, 72, 327879, 6, 126, 125, 84, 196670, 91, 126, 131321, 118, 131320, 127, 262205,
    6, 128, 63, 327874, 6, 129, 128, 109, 327879, 6, 130, 129, 84, 196670, 82, 130, 262205, 6, 131,
    72, 327879, 6, 132, 131, 84, 196670, 86, 132, 262205, 6, 133, 72, 327874, 6, 134, 133, 88,
    327879, 6, 135, 134, 84, 196670, 91, 135, 131321, 118, 131320, 118, 131321, 101, 131320, 101,
    131321, 81, 131320, 81, 262205, 6, 140, 82, 262256, 136, 141, 140, 327816, 136, 143, 141, 142,
    262205, 6, 144, 86, 262256, 136, 145, 144, 327816, 136, 146, 145, 142, 262205, 6, 147, 91,
    262256, 136, 148, 147, 327816, 136, 149, 148, 142, 458832, 137, 151, 143, 146, 149, 150,
    196670, 139, 151, 262205, 152, 155, 154, 262205, 6, 156, 8, 262268, 25, 157, 156, 262205, 6,
    158, 16, 262268, 25, 159, 158, 327760, 160, 161, 157, 159, 262205, 137, 162, 139, 262243, 155,
    161, 162, 65789, 65592,
];

/// Pre-compiled SPIR-V binary bytecode for RGBA8888 -> RGB888 compute repack kernel.
///
/// ```glsl
/// #version 450
/// layout(local_size_x = 64) in;
/// layout(set = 0, binding = 0, rgba8) uniform readonly image2D src_image;
/// layout(set = 0, binding = 1) writeonly buffer DstBuffer { uint dst_data[]; };
/// layout(push_constant) uniform PushConsts { uint width; uint height; };
///
/// void main() {
///     uint chunk_idx = gl_GlobalInvocationID.x; // 4 pixels per chunk
///     uint total_pixels = width * height;
///     uint base_pixel = chunk_idx * 4;
///     if (base_pixel >= total_pixels) return;
///     uint r[4], g[4], b[4];
///     for (uint i = 0; i < 4; ++i) {
///         uint p = base_pixel + i;
///         if (p < total_pixels) {
///             uint px = p % width;
///             uint py = p / width;
///             vec4 col = imageLoad(src_image, ivec2(px, py));
///             r[i] = uint(clamp(col.r * 255.0 + 0.5, 0.0, 255.0));
///             g[i] = uint(clamp(col.g * 255.0 + 0.5, 0.0, 255.0));
///             b[i] = uint(clamp(col.b * 255.0 + 0.5, 0.0, 255.0));
///         } else {
///             r[i] = 0; g[i] = 0; b[i] = 0;
///         }
///     }
///     uint w0 = r[0] | (g[0] << 8) | (b[0] << 16) | (r[1] << 24);
///     uint w1 = g[1] | (b[1] << 8) | (r[2] << 16) | (g[2] << 24);
///     uint w2 = b[2] | (r[3] << 8) | (g[3] << 16) | (b[3] << 24);
///     uint out_word_base = chunk_idx * 3;
///     dst_data[out_word_base + 0] = w0;
///     if (base_pixel + 1 < total_pixels) dst_data[out_word_base + 1] = w1;
///     if (base_pixel + 2 < total_pixels) dst_data[out_word_base + 2] = w2;
/// }
/// ```
pub const REPACK_RGB888_COMP_SPV: &[u32] = &[
    119734787, 65536, 524299, 211, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 64, 1, 1, 196611, 2, 450, 262149,
    4, 1852399981, 0, 327685, 8, 1853188195, 1684627307, 120, 524293, 11, 1197436007, 1633841004,
    1986939244, 1952539503, 1231974249, 68, 393221, 16, 1635020660, 1768972140, 1936483704, 0,
    327685, 17, 1752397136, 1936617283, 29556, 327686, 17, 0, 1952737655, 104, 327686, 17, 1,
    1734960488, 29800, 196613, 19, 0, 327685, 29, 1702060386, 2020175967, 27749, 196613, 40, 105,
    196613, 48, 112, 196613, 57, 30832, 196613, 62, 31088, 196613, 70, 7106403, 327685, 73,
    1600352883, 1734438249, 101, 196613, 84, 114, 196613, 97, 103, 196613, 107, 98, 196613, 126,
    12407, 196613, 144, 12663, 196613, 160, 12919, 393221, 176, 1601467759, 1685221239, 1935762015,
    101, 327685, 181, 1114927940, 1701209717, 114, 393222, 181, 0, 1601467236, 1635017060, 0,
    196613, 183, 0, 262215, 11, 11, 28, 196679, 17, 2, 327752, 17, 0, 35, 0, 327752, 17, 1, 35, 4,
    196679, 73, 24, 262215, 73, 33, 0, 262215, 73, 34, 0, 262215, 180, 6, 4, 196679, 181, 3,
    262216, 181, 0, 25, 327752, 181, 0, 35, 0, 196679, 183, 25, 262215, 183, 33, 1, 262215, 183,
    34, 0, 262215, 210, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167,
    9, 6, 3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 262174, 17,
    6, 6, 262176, 18, 9, 17, 262203, 18, 19, 9, 262165, 20, 32, 1, 262187, 20, 21, 0, 262176, 22,
    9, 6, 262187, 20, 25, 1, 262187, 6, 31, 4, 131092, 35, 196630, 67, 32, 262167, 68, 67, 4,
    262176, 69, 7, 68, 589849, 71, 67, 1, 0, 0, 0, 2, 4, 262176, 72, 0, 71, 262203, 72, 73, 0,
    262167, 79, 20, 2, 262172, 82, 6, 31, 262176, 83, 7, 82, 262176, 86, 7, 67, 262187, 67, 89,
    1132396544, 262187, 67, 91, 1056964608, 262187, 67, 93, 0, 262187, 6, 99, 1, 262187, 6, 109, 2,
    262187, 20, 131, 8, 262187, 20, 136, 16, 262187, 20, 141, 24, 262187, 20, 151, 2, 262187, 20,
    163, 3, 262187, 6, 178, 3, 196637, 180, 6, 196638, 181, 180, 262176, 182, 2, 181, 262203, 182,
    183, 2, 262176, 187, 2, 6, 262187, 6, 209, 64, 393260, 9, 210, 209, 99, 99, 327734, 2, 4, 0, 3,
    131320, 5, 262203, 7, 8, 7, 262203, 7, 16, 7, 262203, 7, 29, 7, 262203, 7, 40, 7, 262203, 7,
    48, 7, 262203, 7, 57, 7, 262203, 7, 62, 7, 262203, 69, 70, 7, 262203, 83, 84, 7, 262203, 83,
    97, 7, 262203, 83, 107, 7, 262203, 7, 126, 7, 262203, 7, 144, 7, 262203, 7, 160, 7, 262203, 7,
    176, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8, 15, 327745, 22, 23, 19, 21,
    262205, 6, 24, 23, 327745, 22, 26, 19, 25, 262205, 6, 27, 26, 327812, 6, 28, 24, 27, 196670,
    16, 28, 262205, 6, 30, 8, 327812, 6, 32, 30, 31, 196670, 29, 32, 262205, 6, 33, 29, 262205, 6,
    34, 16, 327854, 35, 36, 33, 34, 196855, 38, 0, 262394, 36, 37, 38, 131320, 37, 65789, 131320,
    38, 196670, 40, 12, 131321, 41, 131320, 41, 262390, 43, 44, 0, 131321, 45, 131320, 45, 262205,
    6, 46, 40, 327856, 35, 47, 46, 31, 262394, 47, 42, 43, 131320, 42, 262205, 6, 49, 29, 262205,
    6, 50, 40, 327808, 6, 51, 49, 50, 196670, 48, 51, 262205, 6, 52, 48, 262205, 6, 53, 16, 327856,
    35, 54, 52, 53, 196855, 56, 0, 262394, 54, 55, 117, 131320, 55, 262205, 6, 58, 48, 327745, 22,
    59, 19, 21, 262205, 6, 60, 59, 327817, 6, 61, 58, 60, 196670, 57, 61, 262205, 6, 63, 48,
    327745, 22, 64, 19, 21, 262205, 6, 65, 64, 327814, 6, 66, 63, 65, 196670, 62, 66, 262205, 71,
    74, 73, 262205, 6, 75, 57, 262268, 20, 76, 75, 262205, 6, 77, 62, 262268, 20, 78, 77, 327760,
    79, 80, 76, 78, 327778, 68, 81, 74, 80, 196670, 70, 81, 262205, 6, 85, 40, 327745, 86, 87, 70,
    12, 262205, 67, 88, 87, 327813, 67, 90, 88, 89, 327809, 67, 92, 90, 91, 524300, 67, 94, 1, 43,
    92, 93, 89, 262253, 6, 95, 94, 327745, 7, 96, 84, 85, 196670, 96, 95, 262205, 6, 98, 40,
    327745, 86, 100, 70, 99, 262205, 67, 101, 100, 327813, 67, 102, 101, 89, 327809, 67, 103, 102,
    91, 524300, 67, 104, 1, 43, 103, 93, 89, 262253, 6, 105, 104, 327745, 7, 106, 97, 98, 196670,
    106, 105, 262205, 6, 108, 40, 327745, 86, 110, 70, 109, 262205, 67, 111, 110, 327813, 67, 112,
    111, 89, 327809, 67, 113, 112, 91, 524300, 67, 114, 1, 43, 113, 93, 89, 262253, 6, 115, 114,
    327745, 7, 116, 107, 108, 196670, 116, 115, 131321, 56, 131320, 117, 262205, 6, 118, 40,
    327745, 7, 119, 84, 118, 196670, 119, 12, 262205, 6, 120, 40, 327745, 7, 121, 97, 120, 196670,
    121, 12, 262205, 6, 122, 40, 327745, 7, 123, 107, 122, 196670, 123, 12, 131321, 56, 131320, 56,
    131321, 44, 131320, 44, 262205, 6, 124, 40, 327808, 6, 125, 124, 25, 196670, 40, 125, 131321,
    41, 131320, 43, 327745, 7, 127, 84, 21, 262205, 6, 128, 127, 327745, 7, 129, 97, 21, 262205, 6,
    130, 129, 327876, 6, 132, 130, 131, 327877, 6, 133, 128, 132, 327745, 7, 134, 107, 21, 262205,
    6, 135, 134, 327876, 6, 137, 135, 136, 327877, 6, 138, 133, 137, 327745, 7, 139, 84, 25,
    262205, 6, 140, 139, 327876, 6, 142, 140, 141, 327877, 6, 143, 138, 142, 196670, 126, 143,
    327745, 7, 145, 97, 25, 262205, 6, 146, 145, 327745, 7, 147, 107, 25, 262205, 6, 148, 147,
    327876, 6, 149, 148, 131, 327877, 6, 150, 146, 149, 327745, 7, 152, 84, 151, 262205, 6, 153,
    152, 327876, 6, 154, 153, 136, 327877, 6, 155, 150, 154, 327745, 7, 156, 97, 151, 262205, 6,
    157, 156, 327876, 6, 158, 157, 141, 327877, 6, 159, 155, 158, 196670, 144, 159, 327745, 7, 161,
    107, 151, 262205, 6, 162, 161, 327745, 7, 164, 84, 163, 262205, 6, 165, 164, 327876, 6, 166,
    165, 131, 327877, 6, 167, 162, 166, 327745, 7, 168, 97, 163, 262205, 6, 169, 168, 327876, 6,
    170, 169, 136, 327877, 6, 171, 167, 170, 327745, 7, 172, 107, 163, 262205, 6, 173, 172, 327876,
    6, 174, 173, 141, 327877, 6, 175, 171, 174, 196670, 160, 175, 262205, 6, 177, 8, 327812, 6,
    179, 177, 178, 196670, 176, 179, 262205, 6, 184, 176, 327808, 6, 185, 184, 12, 262205, 6, 186,
    126, 393281, 187, 188, 183, 21, 185, 196670, 188, 186, 262205, 6, 189, 29, 327808, 6, 190, 189,
    99, 262205, 6, 191, 16, 327856, 35, 192, 190, 191, 196855, 194, 0, 262394, 192, 193, 194,
    131320, 193, 262205, 6, 195, 176, 327808, 6, 196, 195, 99, 262205, 6, 197, 144, 393281, 187,
    198, 183, 21, 196, 196670, 198, 197, 131321, 194, 131320, 194, 262205, 6, 199, 29, 327808, 6,
    200, 199, 109, 262205, 6, 201, 16, 327856, 35, 202, 200, 201, 196855, 204, 0, 262394, 202, 203,
    204, 131320, 203, 262205, 6, 205, 176, 327808, 6, 206, 205, 109, 262205, 6, 207, 160, 393281,
    187, 208, 183, 21, 206, 196670, 208, 207, 131321, 204, 131320, 204, 65789, 65592,
];

/// Encapsulates GPU Compute pipelines and layouts for fast RGB888 <-> RGBA8888 conversion.
pub struct VulkanRgbCompute {
    ctx: Arc<VulkanContext>,
    unpack_desc_layout: vk::DescriptorSetLayout,
    unpack_pipe_layout: vk::PipelineLayout,
    unpack_shader_module: vk::ShaderModule,
    unpack_pipeline: vk::Pipeline,
    repack_desc_layout: vk::DescriptorSetLayout,
    repack_pipe_layout: vk::PipelineLayout,
    repack_shader_module: vk::ShaderModule,
    repack_pipeline: vk::Pipeline,
}

impl VulkanRgbCompute {
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self> {
        let device = &ctx.device;

        // 1. Unpack pipeline: Binding 0 = STORAGE_BUFFER (src), Binding 1 = STORAGE_IMAGE (dst)
        let unpack_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
        ];
        let unpack_dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: unpack_bindings.len() as u32,
            p_bindings: unpack_bindings.as_ptr(),
            ..Default::default()
        };
        let unpack_desc_layout = unsafe {
            device
                .create_descriptor_set_layout(&unpack_dsl_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create unpack descriptor layout: {e}"
                    ))
                })?
        };

        let push_const_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 8, // width (u32), height (u32)
        };

        let unpack_pl_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &unpack_desc_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_const_range,
            ..Default::default()
        };
        let unpack_pipe_layout = unsafe {
            device
                .create_pipeline_layout(&unpack_pl_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create unpack pipeline layout: {e}"
                    ))
                })?
        };

        let unpack_sm_info = vk::ShaderModuleCreateInfo {
            code_size: std::mem::size_of_val(UNPACK_RGB888_COMP_SPV),
            p_code: UNPACK_RGB888_COMP_SPV.as_ptr(),
            ..Default::default()
        };
        let unpack_shader_module = unsafe {
            device
                .create_shader_module(&unpack_sm_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create unpack shader module: {e}"
                    ))
                })?
        };

        let main_name = c"main";
        let unpack_stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: unpack_shader_module,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let unpack_cp_info = vk::ComputePipelineCreateInfo {
            stage: unpack_stage,
            layout: unpack_pipe_layout,
            ..Default::default()
        };
        let unpack_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[unpack_cp_info], None)
                .map_err(|(_, e)| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create compute unpack pipeline: {e:?}"
                    ))
                })?[0]
        };

        // 2. Repack pipeline: Binding 0 = STORAGE_IMAGE (src), Binding 1 = STORAGE_BUFFER (dst)
        let repack_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
        ];
        let repack_dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: repack_bindings.len() as u32,
            p_bindings: repack_bindings.as_ptr(),
            ..Default::default()
        };
        let repack_desc_layout = unsafe {
            device
                .create_descriptor_set_layout(&repack_dsl_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create repack descriptor layout: {e}"
                    ))
                })?
        };

        let repack_pl_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &repack_desc_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_const_range,
            ..Default::default()
        };
        let repack_pipe_layout = unsafe {
            device
                .create_pipeline_layout(&repack_pl_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create repack pipeline layout: {e}"
                    ))
                })?
        };

        let repack_sm_info = vk::ShaderModuleCreateInfo {
            code_size: std::mem::size_of_val(REPACK_RGB888_COMP_SPV),
            p_code: REPACK_RGB888_COMP_SPV.as_ptr(),
            ..Default::default()
        };
        let repack_shader_module = unsafe {
            device
                .create_shader_module(&repack_sm_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create repack shader module: {e}"
                    ))
                })?
        };

        let repack_stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: repack_shader_module,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let repack_cp_info = vk::ComputePipelineCreateInfo {
            stage: repack_stage,
            layout: repack_pipe_layout,
            ..Default::default()
        };
        let repack_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[repack_cp_info], None)
                .map_err(|(_, e)| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create compute repack pipeline: {e:?}"
                    ))
                })?[0]
        };

        Ok(Self {
            ctx,
            unpack_desc_layout,
            unpack_pipe_layout,
            unpack_shader_module,
            unpack_pipeline,
            repack_desc_layout,
            repack_pipe_layout,
            repack_shader_module,
            repack_pipeline,
        })
    }

    /// Records a GPU compute command to unpack packed RGB888 storage buffer into an RGBA8888 image.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_unpack_rgb888(
        &self,
        cmd_buf: vk::CommandBuffer,
        src_buf: vk::Buffer,
        src_buf_size: vk::DeviceSize,
        dst_image: vk::Image,
        dst_view: vk::ImageView,
        width: u32,
        height: u32,
        post_layout: vk::ImageLayout,
        post_access: vk::AccessFlags,
        post_stage: vk::PipelineStageFlags,
    ) -> Result<vk::DescriptorPool> {
        let device = &self.ctx.device;

        // 1. Create a transient descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let desc_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create compute unpack desc pool: {e}"
                    ))
                })?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.unpack_desc_layout,
            ..Default::default()
        };
        let desc_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                device.destroy_descriptor_pool(desc_pool, None);
                ScalixError::ExecutionFailed(format!(
                    "Failed to allocate unpack descriptor set: {e}"
                ))
            })?[0]
        };

        let buf_info = vk::DescriptorBufferInfo {
            buffer: src_buf,
            offset: 0,
            range: src_buf_size,
        };
        let img_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: dst_view,
            sampler: vk::Sampler::null(),
        };

        let write_sets = [
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buf_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &img_info,
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&write_sets, &[]) };

        // 2. Transition dst_image: UNDEFINED -> GENERAL
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let barrier_pre = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            image: dst_image,
            subresource_range,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_pre],
            );
        }

        // 3. Bind Compute Pipeline & Dispatch
        let push_consts = [width, height];
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                push_consts.as_ptr() as *const u8,
                std::mem::size_of_val(&push_consts),
            )
        };

        unsafe {
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.unpack_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.unpack_pipe_layout,
                0,
                &[desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd_buf,
                self.unpack_pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(cmd_buf, width.div_ceil(16), height.div_ceil(16), 1);
        }

        // 4. Transition dst_image: GENERAL -> post_layout
        let barrier_post = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: post_layout,
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: post_access,
            image: dst_image,
            subresource_range,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                post_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_post],
            );
        }

        Ok(desc_pool)
    }

    /// Records a GPU compute command to repack an RGBA8888 image into a packed RGB888 storage buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_repack_rgb888(
        &self,
        cmd_buf: vk::CommandBuffer,
        src_image: vk::Image,
        src_view: vk::ImageView,
        dst_buf: vk::Buffer,
        dst_buf_size: vk::DeviceSize,
        width: u32,
        height: u32,
        current_layout: vk::ImageLayout,
        current_access: vk::AccessFlags,
        current_stage: vk::PipelineStageFlags,
    ) -> Result<vk::DescriptorPool> {
        let device = &self.ctx.device;

        // 1. Create a transient descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let desc_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create compute repack desc pool: {e}"
                    ))
                })?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.repack_desc_layout,
            ..Default::default()
        };
        let desc_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                device.destroy_descriptor_pool(desc_pool, None);
                ScalixError::ExecutionFailed(format!(
                    "Failed to allocate repack descriptor set: {e}"
                ))
            })?[0]
        };

        let img_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: src_view,
            sampler: vk::Sampler::null(),
        };
        let buf_info = vk::DescriptorBufferInfo {
            buffer: dst_buf,
            offset: 0,
            range: dst_buf_size,
        };

        let write_sets = [
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &img_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buf_info,
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&write_sets, &[]) };

        // 2. Transition src_image: current_layout -> GENERAL
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let barrier_pre = vk::ImageMemoryBarrier {
            old_layout: current_layout,
            new_layout: vk::ImageLayout::GENERAL,
            src_access_mask: current_access,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            image: src_image,
            subresource_range,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                current_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_pre],
            );
        }

        // 3. Bind Repack Pipeline & Dispatch
        let push_consts = [width, height];
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                push_consts.as_ptr() as *const u8,
                std::mem::size_of_val(&push_consts),
            )
        };

        let total_pixels = width * height;
        let total_chunks = total_pixels.div_ceil(4);
        let group_x = total_chunks.div_ceil(64);

        unsafe {
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.repack_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.repack_pipe_layout,
                0,
                &[desc_set],
                &[],
            );
            device.cmd_push_constants(
                cmd_buf,
                self.repack_pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(cmd_buf, group_x, 1, 1);
        }

        // 4. Memory barrier for dst_buf: COMPUTE_SHADER write -> HOST read
        let buf_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::HOST_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: dst_buf,
            offset: 0,
            size: dst_buf_size,
            ..Default::default()
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[buf_barrier],
                &[],
            );
        }

        Ok(desc_pool)
    }
}

impl Drop for VulkanRgbCompute {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.destroy_pipeline(self.unpack_pipeline, None);
            device.destroy_shader_module(self.unpack_shader_module, None);
            device.destroy_pipeline_layout(self.unpack_pipe_layout, None);
            device.destroy_descriptor_set_layout(self.unpack_desc_layout, None);

            device.destroy_pipeline(self.repack_pipeline, None);
            device.destroy_shader_module(self.repack_shader_module, None);
            device.destroy_pipeline_layout(self.repack_pipe_layout, None);
            device.destroy_descriptor_set_layout(self.repack_desc_layout, None);
        }
    }
}
