CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -O2
INCLUDES ?= -I../include
LIBDIR ?= ../target/debug
LDFLAGS ?= -L$(LIBDIR) -lscalix -ljpeg -lpthread -Wl,-rpath,'$$ORIGIN/$(LIBDIR)'

TARGETS = cpp_basic cpp_jpeg cpp_benchmark

.PHONY: all clean run benchmark cargo-build

all: cargo-build $(TARGETS)

cargo-build:
	@echo "[Building Scalix Cargo Workspace]"
	cargo build --manifest-path ../Cargo.toml

cpp_basic: cpp_basic.cpp
	@echo "[Compiling $@]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

cpp_jpeg: cpp_jpeg.cpp
	@echo "[Compiling $@ with libjpeg-turbo]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

OPENCV_FLAGS ?= $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null || echo "")
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null || echo "")

cpp_benchmark: cpp_benchmark.cpp
	@echo "[Compiling $@ with OpenCV comparison support]"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $< $(LDFLAGS) $(OPENCV_LIBS) -o $@

run: cargo-build cpp_basic cpp_jpeg
	@echo "\n=== Running cpp_basic ==="
	./cpp_basic
	@echo "\n=== Running cpp_jpeg ==="
	./cpp_jpeg ../assets/sample.jpg /tmp/output_sample.jpg

benchmark: cargo-build cpp_benchmark
	@echo "\n=== Running cpp_benchmark ==="
	./cpp_benchmark

clean:
	rm -f $(TARGETS) /tmp/output_sample.jpg
