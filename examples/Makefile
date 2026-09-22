CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -O2
INCLUDES ?= -I../include -I.
LIBDIR ?= ../target/debug
LDFLAGS ?= -L$(LIBDIR) -lscalix -ljpeg -lpthread -Wl,-rpath,'$$ORIGIN/$(LIBDIR)'

OPENCV_FLAGS ?= $(shell (pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null || echo "") | sed 's/-I/-isystem /g')
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null || echo "")

COMMON_OBJS = jpeg_io.o benchmark_utils.o
TARGETS = cpp_basic cpp_jpeg cpp_benchmark

.PHONY: all clean run benchmark cargo-build

all: cargo-build $(TARGETS)

cargo-build:
	@echo "[Building Scalix Cargo Workspace]"
	cargo build --manifest-path ../Cargo.toml

jpeg_io.o: jpeg_io.cpp jpeg_io.hpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

benchmark_utils.o: benchmark_utils.cpp benchmark_utils.hpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

cpp_basic: cpp_basic.cpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

cpp_jpeg: cpp_jpeg.cpp jpeg_io.o
	@echo "[Compiling $@ with libjpeg-turbo]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

cpp_benchmark: cpp_benchmark.cpp jpeg_io.o benchmark_utils.o
	@echo "[Compiling $@ with OpenCV comparison support]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $^ $(LDFLAGS) $(OPENCV_LIBS) -o $@

run: cargo-build cpp_basic cpp_jpeg
	@echo "\n=== Running cpp_basic ==="
	./cpp_basic
	@echo "\n=== Running cpp_jpeg ==="
	./cpp_jpeg ../assets/sample.jpg /tmp/output_sample.jpg

benchmark: cargo-build cpp_benchmark
	@echo "\n=== Running cpp_benchmark ==="
	./cpp_benchmark

clean:
	rm -f $(TARGETS) $(COMMON_OBJS) /tmp/output_sample.jpg /tmp/bench_sample_*.jpg
