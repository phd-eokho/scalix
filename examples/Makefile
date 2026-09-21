CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -O2
INCLUDES ?= -I../include
LIBDIR ?= ../target/debug
LDFLAGS ?= -L$(LIBDIR) -lscalix -ljpeg -lpthread -Wl,-rpath,'$$ORIGIN/$(LIBDIR)'

TARGETS = cpp_basic cpp_jpeg

.PHONY: all clean run cargo-build

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

run: all
	@echo "\n=== Running cpp_basic ==="
	./cpp_basic
	@echo "\n=== Running cpp_jpeg ==="
	./cpp_jpeg ../assets/sample.jpg /tmp/output_sample.jpg

clean:
	rm -f $(TARGETS) /tmp/output_sample.jpg
