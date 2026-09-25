CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -O2
INCLUDES ?= -I../include -I.
LIBDIR ?= ../target/debug
LDFLAGS ?= -L$(LIBDIR) -lscalix -ljpeg -lpthread -Wl,-rpath,'$$ORIGIN/$(LIBDIR)'

OPENCV_FLAGS ?= $(shell (pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null || echo "") | sed 's/-I/-isystem /g')
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null || echo "")

COMMON_OBJS = jpeg_io.o benchmark_utils.o
TARGETS = cpp_basic cpp_jpeg cpp_benchmark_vulkan cpp_benchmark_gl cpp_benchmark_opencl

.PHONY: all clean run benchmark benchmark_vulkan benchmark_gl benchmark_opencl cargo-build

all: cargo-build $(TARGETS)

cargo-build:
	@echo "[Building Scalix Cargo Workspace]"
	cargo build --manifest-path ../Cargo.toml

jpeg_io.o: jpeg_io.cpp jpeg_io.hpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

benchmark_utils.o: benchmark_utils.cpp benchmark_utils.hpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) -c $< -o $@

cpp_basic: cpp_basic.cpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

cpp_jpeg: cpp_jpeg.cpp jpeg_io.o
	@echo "[Compiling $@ with libjpeg-turbo]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

cpp_benchmark_vulkan: cpp_benchmark_vulkan.cpp jpeg_io.o benchmark_utils.o
	@echo "[Compiling $@ with OpenCV comparison support (Vulkan)]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $^ $(LDFLAGS) $(OPENCV_LIBS) -o $@

cpp_benchmark_gl: cpp_benchmark_gl.cpp jpeg_io.o benchmark_utils.o
	@echo "[Compiling $@ with OpenCV comparison support (OpenGL/EGL)]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $^ $(LDFLAGS) $(OPENCV_LIBS) -o $@

cpp_benchmark_opencl: cpp_benchmark_opencl.cpp jpeg_io.o benchmark_utils.o
	@echo "[Compiling $@ with OpenCV comparison support (OpenCL)]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $^ $(LDFLAGS) $(OPENCV_LIBS) -o $@

run: cargo-build cpp_basic cpp_jpeg
	@echo "\n=== Running cpp_basic ==="
	./cpp_basic
	@echo "\n=== Running cpp_jpeg ==="
	./cpp_jpeg ../assets/sample.jpg /tmp/output_sample.jpg

benchmark: benchmark_vulkan benchmark_gl benchmark_opencl

benchmark_vulkan: cargo-build cpp_benchmark_vulkan
	@echo "\n=== Running cpp_benchmark_vulkan (Vulkan) ==="
	./cpp_benchmark_vulkan

benchmark_gl: cargo-build cpp_benchmark_gl
	@echo "\n=== Running cpp_benchmark_gl (OpenGL/EGL) ==="
	./cpp_benchmark_gl

benchmark_opencl: cargo-build cpp_benchmark_opencl
	@echo "\n=== Running cpp_benchmark_opencl (OpenCL) ==="
	./cpp_benchmark_opencl

clean:
	rm -f $(TARGETS) cpp_benchmark $(COMMON_OBJS) /tmp/output_sample.jpg /tmp/bench_sample_*.jpg /tmp/bench_vulkan_sample_*.jpg /tmp/bench_gl_sample_*.jpg /tmp/bench_opencl_sample_*.jpg
