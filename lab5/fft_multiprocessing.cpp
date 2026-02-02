#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <stdexcept>
#include <omp.h>
#include <cmath> 
#include <chrono>



std::vector<std::complex<double>> fft_radix2_tasked(const std::vector<std::complex<double>>& x, int depth, int maxDepth)
{
    const std::size_t n = x.size();

    if (n == 0) return {};
    if (n == 1) return { x[0] };

    std::vector<std::complex<double>> xe(n/2), xo(n/2);
    for (std::size_t j = 0; j < n/2; ++j) {
        xe[j] = x[2*j];
        xo[j] = x[2*j + 1];
    }

    std::vector<std::complex<double>> T1, T2;

    if (depth < maxDepth) {
        #pragma omp task shared(T1) firstprivate(xe, depth, maxDepth)
        {
            T1 = fft_radix2_tasked(xe, depth + 1, maxDepth);
        }
        #pragma omp task shared(T2) firstprivate(xo, depth, maxDepth)
        {
            T2 = fft_radix2_tasked(xo, depth + 1, maxDepth);
        }
        #pragma omp taskwait
    } else {
        T1 = fft_radix2_tasked(xe, depth + 1, maxDepth);
        T2 = fft_radix2_tasked(xo, depth + 1, maxDepth);
    }

    std::vector<std::complex<double>> F(n);
    const double pi = std::numbers::pi_v<double>;
    const std::complex<double> wlen = std::polar(1.0, -2.0 * pi / static_cast<double>(n));
    std::complex<double> w = 1.0;
    for (std::size_t k = 0; k < n/2; ++k) {
        const std::complex<double> t = w * T2[k];
        F[k]        = T1[k] + t;
        F[k + n/2]  = T1[k] - t;
        w *= wlen;
    }
    return F;
}

std::vector<std::complex<double>> ifft_radix2_tasked(const std::vector<std::complex<double>>& X, int depth, int maxDepth)
{
    const std::size_t n = X.size();
    if (n == 0) return {};
    if (n == 1) return { X[0] };

    std::vector<std::complex<double>> Xe(n/2), Xo(n/2);
    for (std::size_t j = 0; j < n/2; ++j) {
        Xe[j] = X[2*j];
        Xo[j] = X[2*j + 1];
    }

    std::vector<std::complex<double>> t1, t2;

    if (depth < maxDepth) {
        #pragma omp task shared(t1) firstprivate(Xe, depth, maxDepth)
        {
            t1 = ifft_radix2_tasked(Xe, depth + 1, maxDepth);
        }
        #pragma omp task shared(t2) firstprivate(Xo, depth, maxDepth)
        {
            t2 = ifft_radix2_tasked(Xo, depth + 1, maxDepth);
        }
        #pragma omp taskwait
    } else {
        t1 = ifft_radix2_tasked(Xe, depth + 1, maxDepth);
        t2 = ifft_radix2_tasked(Xo, depth + 1, maxDepth);
    }

    std::vector<std::complex<double>> x(n);
    const double pi = std::numbers::pi_v<double>;
    const std::complex<double> wlen = std::polar(1.0, +2.0 * pi / static_cast<double>(n));
    std::complex<double> w = 1.0;
    for (std::size_t k = 0; k < n/2; ++k) {
        const std::complex<double> t = w * t2[k];
        x[k]        = t1[k] + t;
        x[k + n/2]  = t1[k] - t;
        w *= wlen;
    }
    for (auto& v : x) v *= 0.5;
    return x;
}

void test_fft_multiprocessing() {
    using std::cout;

    std::cout << std::fixed << std::setprecision(6);

    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());

    // Подберём разумную глубину порождения задач (≈ 2 * log2(потоков))
    int maxDepth = 2 * static_cast<int>(std::log2(std::max(1, omp_get_max_threads())));

    // Пример: реальный сигнал N=8
    const std::size_t N = 8;
    std::vector<std::complex<double>> x = {1,2,3,4,5,6,7,8};
    std::vector<std::complex<double>> X, x_rec;

    // Параллельный FFT
    #pragma omp parallel
    {
        #pragma omp single
        {
            X = fft_radix2_tasked(x, 0, maxDepth);
        }
    }

    // Параллельный IFFT
    #pragma omp parallel
    {
        #pragma omp single
        {
            x_rec = ifft_radix2_tasked(X, 0, maxDepth);
        }
    }

    cout << "x:           "; for (auto v : x)     cout << v << " "; cout << "\n";
    cout << "FFT(x):      "; for (auto v : X)     cout << v << " "; cout << "\n";
    cout << "IFFT(FFT(x)):"; for (auto v : x_rec) cout << v << " "; cout << "\n";

    double max_err = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(x_rec[i] - x[i]));
    cout << "Макс. ошибка восстановления: " << max_err << "\n";

}

std::chrono::duration<double> test_fft_multiprocessing_time(std::vector<std::complex<double>> elements) {
    auto time_start = std::chrono::steady_clock::now(); 
    int maxDepth = 2 * static_cast<int>(std::log2(std::max(1, omp_get_max_threads())));
    std::vector<std::complex<double>> X, x_rec;
    #pragma omp parallel
    {
        #pragma omp single
        {
            X = fft_radix2_tasked(elements, 0, maxDepth);
        }
    }
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    return elapsed;
}
