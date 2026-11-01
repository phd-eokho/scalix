// Separable high-order resampling kernels shared by raster and compute paths.
// Coordinate convention matches the Vulkan compute kernels:
//   src = (dst + 0.5) * scale - 0.5, edges clamped.

const float PI = 3.14159265358979323846;

vec4 fetchClamped(sampler2D tex, ivec2 p, ivec2 size) {
    return texelFetch(tex, clamp(p, ivec2(0), size - ivec2(1)), 0);
}

// Catmull-Rom cubic convolution (Keys, 1981, a = -0.5), 4x4 taps.
void catmullRomWeights(float f, out float w[4]) {
    float f2 = f * f;
    float f3 = f2 * f;
    w[0] = -0.5 * f3 + f2 - 0.5 * f;
    w[1] = 1.5 * f3 - 2.5 * f2 + 1.0;
    w[2] = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    w[3] = 0.5 * f3 - 0.5 * f2;
}

vec4 resampleCatmullRom(sampler2D tex, vec2 dstPixel, vec2 scale) {
    ivec2 size = textureSize(tex, 0);
    vec2 uv = (dstPixel + 0.5) * scale - 0.5;
    vec2 base = floor(uv);
    vec2 f = uv - base;
    ivec2 b = ivec2(base);

    float wx[4];
    float wy[4];
    catmullRomWeights(f.x, wx);
    catmullRomWeights(f.y, wy);

    vec4 col = vec4(0.0);
    for (int j = 0; j < 4; ++j) {
        vec4 row = vec4(0.0);
        for (int i = 0; i < 4; ++i) {
            row += wx[i] * fetchClamped(tex, b + ivec2(i - 1, j - 1), size);
        }
        col += wy[j] * row;
    }
    return clamp(col, 0.0, 1.0);
}

// Lanczos-3 windowed sinc, 6x6 taps, weights normalized to preserve DC.
float lanczos3Weight(float x) {
    float ax = abs(x);
    if (ax < 1e-5) return 1.0;
    if (ax >= 3.0) return 0.0;
    float pix = PI * ax;
    return (sin(pix) * sin(pix / 3.0) * 3.0) / (pix * pix);
}

vec4 resampleLanczos3(sampler2D tex, vec2 dstPixel, vec2 scale) {
    ivec2 size = textureSize(tex, 0);
    vec2 uv = (dstPixel + 0.5) * scale - 0.5;
    vec2 base = floor(uv);
    vec2 f = uv - base;
    ivec2 b = ivec2(base);

    float wx[6];
    float wy[6];
    float sumX = 0.0;
    float sumY = 0.0;
    for (int i = 0; i < 6; ++i) {
        wx[i] = lanczos3Weight(float(i - 2) - f.x);
        wy[i] = lanczos3Weight(float(i - 2) - f.y);
        sumX += wx[i];
        sumY += wy[i];
    }
    float invX = 1.0 / max(abs(sumX), 1e-5);
    float invY = 1.0 / max(abs(sumY), 1e-5);

    vec4 col = vec4(0.0);
    for (int j = 0; j < 6; ++j) {
        vec4 row = vec4(0.0);
        for (int i = 0; i < 6; ++i) {
            row += (wx[i] * invX) * fetchClamped(tex, b + ivec2(i - 2, j - 2), size);
        }
        col += (wy[j] * invY) * row;
    }
    return clamp(col, 0.0, 1.0);
}
