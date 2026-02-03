#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <stdexcept>
#include <omp.h>
#include <cmath> 
#include <chrono>

using namespace std::numbers;


void fft_radix2_core(std::complex<double>* x, size_t n, int depth, int maxDepth, int inverse) {

    if (n == 0) return;
    if (n == 1) return;

    std::vector<std::complex<double>> xe(n/2), xo(n/2);

    #pragma omp parallel for if(depth == 0 && n > 1000)
    for (std::size_t j = 0; j < n/2; ++j) {
        xe[j] = x[2*j];
        xo[j] = x[2*j + 1];
    }


    #pragma omp task shared(xe) if(depth < maxDepth && n > 1000)
        {
            fft_radix2_core(xe.data(), n / 2, depth + 1, maxDepth, 1);
        }
    #pragma omp task shared(xo) if(depth < maxDepth && n > 1000)
        {
            fft_radix2_core(xo.data(), n / 2, depth + 1, maxDepth, 1);
        }
    #pragma omp taskwait


    #pragma omp parallel for if(depth == 0 && n > 1000)
    for (size_t i = 0; i < n / 2; i++) {
        double angle = (inverse == 1) ? -2.0 * pi_v<double> * i / n : 2.0 * pi_v<double> * i / n;
        std::complex<double> w = std::polar(1.0, angle);

        std::complex<double> t = w * xe[i];
        x[i] = xe[i] + t;
        x[i + n / 2] = xo[i] - t;
    }
}

void fft_radix2_recursive(std::complex<double>* x, size_t n, int maxDepth, int inverse) {
    #pragma omp parallel
    #pragma omp single nowait
    {
        fft_radix2_core(x, n, 0, maxDepth, inverse);
    }
}

void fft_radix2_openmp(std::complex<double>* x, size_t n, int maxDepth) {
    fft_radix2_recursive(x, n, maxDepth, 1);
}

void ifft_radix2_openmp(std::complex<double>* X, size_t n, int maxDepth) {
    fft_radix2_recursive(X, n, maxDepth, -1);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        X[i] /= static_cast<double>(n);
    }
}
