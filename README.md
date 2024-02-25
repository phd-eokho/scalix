# resizer-opengl
An OpenGL image resizer (linear, bicubic)

### How to use
```sh
$ g++ -O3 -std=c++17 -o resizer_gl main.cpp resizer_gl.cpp \
  -lGLESv2 -lEGL -lglut $(pkg-config --cflags --libs opencv4) \
  && ./resizer_gl
```
