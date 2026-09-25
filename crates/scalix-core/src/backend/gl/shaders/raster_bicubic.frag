in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uTexSize;

vec4 cubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

vec4 textureBicubic(sampler2D sampler, vec2 texCoords) {
    vec2 texSize = uTexSize;
    vec2 invTexSize = 1.0 / texSize;
    texCoords = texCoords * texSize - 0.5;

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
    vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

    vec4 sample0 = texture(sampler, vec2(offset.x, offset.z) * invTexSize);
    vec4 sample1 = texture(sampler, vec2(offset.y, offset.z) * invTexSize);
    vec4 sample2 = texture(sampler, vec2(offset.x, offset.w) * invTexSize);
    vec4 sample3 = texture(sampler, vec2(offset.y, offset.w) * invTexSize);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

void main() {
    FragColor = textureBicubic(uTexture, vTexCoord);
}
