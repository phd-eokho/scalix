

#include <opencv2/opencv.hpp>
#include <chrono>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "resizer_gl.h"

using namespace std;
using namespace core::image;

int main(int argc, const char *argv[])
{
    constexpr auto SIZE = 320;
    ResizerEGL egl(SIZE, SIZE);

    int64_t elapsed = 0;
    int N = 100;
    auto m0 = cv::imread("assets/sample.jpg");

    const auto random_roi = [&m0]()
    {
        auto width = m0.size().width;
        auto height = m0.size().height;

        int w_min = width * 0.25;
        int h_min = height * 0.25;

        auto x0 = rand() % (width / 2);
        auto y0 = rand() % (height / 2);
        auto w = max(w_min, rand() % (width - x0));
        auto h = max(h_min, rand() % (height - y0));
        return cv::Rect(x0, y0, w, h);
    };

    auto d0 = cv::Mat(SIZE, SIZE, CV_8UC4);
    {
        auto m = m0;
        cv::Mat m_rgba;
        cv::cvtColor(m, m_rgba, cv::COLOR_BGR2RGBA);
        egl.resize(d0.data, m_rgba.data, m.size().width, m.size().height);

        int dw, dh;
        if (m.size().width > m.size().height)
        {
            dw = SIZE;
            dh = SIZE * (static_cast<float>(m.size().height) / m.size().width);
        }
        else
        {
            dw = SIZE * (static_cast<float>(m.size().width) / m.size().height);
            dh = SIZE;
        }

        cv::Mat d;
        cv::cvtColor(d0(cv::Rect(0, 0, dw, dh)), d, cv::COLOR_RGBA2BGR);

        cv::imwrite("result.jpg", d);
    }

    for (auto i = 0; i < N; ++i)
    {
        auto m = m0(random_roi());
        printf("iter %02d: m w=%d h=%d\n", i, m.size().width, m.size().height);

        cv::Mat m_rgba;
        cv::cvtColor(m, m_rgba, cv::COLOR_BGR2RGBA);

        auto begin = std::chrono::high_resolution_clock::now();
        egl.resize(d0.data, m_rgba.data, m.size().width, m.size().height);
        auto end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        int dw, dh;
        if (m.size().width > m.size().height)
        {
            dw = SIZE;
            dh = SIZE * (static_cast<float>(m.size().height) / m.size().width);
        }
        else
        {
            dw = SIZE * (static_cast<float>(m.size().width) / m.size().height);
            dh = SIZE;
        }

        cv::Mat d;
        cv::cvtColor(d0(cv::Rect(0, 0, dw, dh)), d, cv::COLOR_RGBA2BGR);

        char buf[256];
        snprintf(buf, 256, "result_iter_%d.jpg", i);
        cv::imwrite(buf, d);
    }
    printf("Measured avg. %.3fms\n", elapsed / static_cast<float>(N));

    return EXIT_SUCCESS;
}