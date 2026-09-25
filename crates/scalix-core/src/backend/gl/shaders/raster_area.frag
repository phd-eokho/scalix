in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uTexSize;
uniform vec2 uDstSize;

vec4 sampleArea(sampler2D tex, vec2 fragCoord) {
    vec2 scale = uTexSize / uDstSize;
    vec2 dCoord = floor(fragCoord);

    float x0 = dCoord.x * scale.x;
    float x1 = (dCoord.x + 1.0) * scale.x;
    float y0 = dCoord.y * scale.y;
    float y1 = (dCoord.y + 1.0) * scale.y;

    int sx_min = int(floor(x0));
    int sx_max = int(floor(x1));
    sx_max -= int(float(sx_max) == x1 && sx_max > sx_min);

    int sy_min = int(floor(y0));
    int sy_max = int(floor(y1));
    sy_max -= int(float(sy_max) == y1 && sy_max > sy_min);

    vec4 sum_col = vec4(0.0);
    float total_weight = 0.0;

    for (int sy = sy_min; sy <= sy_max; ++sy) {
        float wy = max(0.0, min(float(sy) + 1.0, y1) - max(float(sy), y0));
        int cy = clamp(sy, 0, int(uTexSize.y) - 1);
        for (int sx = sx_min; sx <= sx_max; ++sx) {
            float wx = max(0.0, min(float(sx) + 1.0, x1) - max(float(sx), x0));
            float w = wx * wy;
            int cx = clamp(sx, 0, int(uTexSize.x) - 1);
            sum_col += texelFetch(tex, ivec2(cx, cy), 0) * w;
            total_weight += w;
        }
    }

    return (total_weight > 0.0) ? (sum_col / total_weight) : texelFetch(tex, ivec2(clamp(sx_min, 0, int(uTexSize.x) - 1), clamp(sy_min, 0, int(uTexSize.y) - 1)), 0);
}

void main() {
    FragColor = sampleArea(uTexture, gl_FragCoord.xy);
}
