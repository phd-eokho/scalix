in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uTexSize;
uniform vec2 uDstSize;

void main() {
    FragColor = resampleLanczos3(uTexture, floor(gl_FragCoord.xy), uTexSize / uDstSize);
}
