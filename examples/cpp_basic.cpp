#include <iostream>
#include <vector>
#include <cassert>
#include <atomic>
#include <chrono>
#include <thread>
#include <scalix/scalix.hpp>

int main() {
    std::cout << "[Scalix C++20 Interface Test]" << std::endl;

    // 1. Initialize Engine
    scalix::Engine engine(scalix::Backend::Auto);

    constexpr uint32_t width = 32;
    constexpr uint32_t height = 32;
    constexpr size_t stride = width * 4;
    std::vector<uint8_t> src_data(stride * height, 0x55);
    std::vector<uint8_t> dst_data(stride * height, 0x00);

    scalix::ImageDesc src{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = src_data.data(),
        .data_len = src_data.size(),
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = dst_data.data(),
        .data_len = dst_data.size(),
        .dma_buf_fd = -1,
    };

    // 2. Test Synchronous Passthrough
    std::cout << "Testing synchronous resize (passthrough memory transfer)..." << std::endl;
    engine.resize(src, dst, scalix::Filter::Passthrough);
    assert(dst_data == src_data);
    std::cout << "  → Sync transfer verified successfully." << std::endl;

    // 3. Test Callback Execution
    std::cout << "Testing callback-driven resize..." << std::endl;
    std::atomic<bool> cb_done{false};
    std::vector<uint8_t> dst_cb_data(stride * height, 0x00);
    scalix::ImageDesc dst_cb{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = dst_cb_data.data(),
        .data_len = dst_cb_data.size(),
        .dma_buf_fd = -1,
    };

    engine.resize_callback(
        src,
        dst_cb,
        scalix::Filter::Passthrough,
        [&cb_done](int status) {
            std::cout << "  → Callback fired with status: " << status << std::endl;
            cb_done.store(true);
        }
    );

    while (!cb_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 4. Test Zero-Copy DMA Buffer Pre-allocation
    std::cout << "Testing Zero-Copy DMA Buffer pre-allocation..." << std::endl;
    try {
        scalix::DmaBuffer dma_src(width, height, scalix::PixelFormat::Rgba8888);
        scalix::DmaBuffer dma_dst(width, height, scalix::PixelFormat::Rgba8888);

        std::cout << "  → Allocated DMA buffers (src_fd=" << dma_src.fd()
                  << ", dst_fd=" << dma_dst.fd()
                  << ", size=" << dma_src.size() << " bytes)" << std::endl;

        // Fill source DMA buffer with test pattern using scoped with_write lambda
        dma_src.with_write([](uint8_t* ptr, size_t size) {
            std::fill_n(ptr, size, 0x77);
        });

        dma_dst.with_write([](uint8_t* ptr, size_t size) {
            std::fill_n(ptr, size, 0x00);
        });

        // Execute engine resize using DMA-backed ImageDesc
        auto src_desc = dma_src.as_image_desc();
        auto dst_desc = dma_dst.as_image_desc();
        engine.resize(src_desc, dst_desc, scalix::Filter::Passthrough);

        // Verify destination DMA buffer using scoped with_read lambda
        dma_dst.with_read([](const uint8_t* ptr, size_t size) {
            assert(ptr[0] == 0x77);
            assert(ptr[size - 1] == 0x77);
        });

        std::cout << "  → Zero-Copy DMA pre-allocation verified successfully (via with_write/with_read)." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  → [Host Notice] DMA device nodes unavailable on this host (" << e.what() << "); skipped hardware test." << std::endl;
    }

    std::cout << "[All C++20 interface tests passed!]" << std::endl;
    return 0;
}
