#include <complex>
#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <algorithm>
#include <stdexcept>
#include <chrono>


std::vector<std::complex<double>> fft_radix2(const std::vector<std::complex<double>>& x)
{
    const std::size_t n = x.size();

    if (n == 0) return {};
    if (n == 1) return { x[0] };

    std::vector<std::complex<double>> xe(n/2), xo(n/2);
    for (std::size_t j = 0; j < n/2; ++j) {
        xe[j] = x[2*j];
        xo[j] = x[2*j + 1];
    }

    auto T1 = fft_radix2(xe);
    auto T2 = fft_radix2(xo);

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

std::vector<std::complex<double>> ifft_radix2(const std::vector<std::complex<double>>& X)
{
    const std::size_t n = X.size();

    if (n == 0) return {};
    // Приём с сопряжением: IFFT(X) = conj( FFT(conj(X)) ) / n
    std::vector<std::complex<double>> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::conj(X[i]);
    tmp = fft_radix2(tmp);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::conj(tmp[i]) / static_cast<double>(n);
    return tmp;
}

// ---------- Ручная круговая свёртка ----------
std::vector<std::complex<double>> circular_convolution_manual(
    const std::vector<std::complex<double>>& x,
    const std::vector<std::complex<double>>& y)
{
    const std::size_t N = x.size();
    std::vector<std::complex<double>> z(N, 0.0);
    for (std::size_t n = 0; n < N; ++n) {
        std::complex<double> s = 0.0;
        for (std::size_t m = 0; m < N; ++m) {
            s += x[m] * y[(n + N - m) % N];
        }
        z[n] = s;
    }
    return z;
}

void test_fft_singleprocessing() {
    std::cout << std::fixed << std::setprecision(6);

    {
        const std::size_t N = 8;
        std::vector<std::complex<double>> x = {1,2,3,4,5,6,7,8};
        auto X = fft_radix2(x);
        auto x_rec = ifft_radix2(X);

        double max_err = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            max_err = std::max(max_err, std::abs(x_rec[i] - x[i]));

        std::cout << "Обратимость FFT/IFFT (N=8): макс. ошибка = " << max_err << "\n\n";
    }

    {
        const std::size_t N = 4;
        std::vector<std::complex<double>> x = {3.0, 4.0, 0.0, 0.0};
        std::vector<std::complex<double>> y = {4.0, 3.0, 0.0, 0.0};

        auto X = fft_radix2(x);
        auto Y = fft_radix2(y);

        std::vector<std::complex<double>> Z(N);
        for (std::size_t k = 0; k < N; ++k) Z[k] = X[k] * Y[k];

        auto z_fft = ifft_radix2(Z);

        auto z_circ = circular_convolution_manual(x, y);

        std::cout << "Круговая свёртка (N=4):\n";
        for (std::size_t n = 0; n < N; ++n) {
            std::cout << "n=" << n
                      << "  z_fft="  << z_fft[n]
                      << "  z_circ=" << z_circ[n] << "\n";
        }

        double max_err = 0.0;
        for (std::size_t n = 0; n < N; ++n)
            max_err = std::max(max_err, std::abs(z_fft[n] - z_circ[n]));
        std::cout << "Макс. расхождение: " << max_err << "\n";
        std::cout << "Ожидаем: [12, 25, 12, 0]\n\n";
    }

    {
        const std::size_t N = 4;
        std::vector<std::complex<double>> x = {3.0, 4.0, 0.0, 0.0};

        std::vector<std::complex<double>> xe = {x[0], x[2]};
        std::vector<std::complex<double>> xo = {x[1], x[3]};
        auto T1 = fft_radix2(xe);
        auto T2 = fft_radix2(xo);

        std::vector<std::complex<double>> F = fft_radix2(x);
        std::cout << "Подсказка преподавателя (N=4):\n";
        std::cout << "T1 (FFT чётных): "; for (auto v : T1) std::cout << v << " "; std::cout << "\n";
        std::cout << "T2 (FFT нечётных): "; for (auto v : T2) std::cout << v << " "; std::cout << "\n";
        std::cout << "F (полный FFT):   "; for (auto v : F)  std::cout << v << " "; std::cout << "\n";
    }

}


std::chrono::duration<double> test_fft_time(std::vector<std::complex<double>> elements) {
    auto time_start = std::chrono::steady_clock::now(); 
    fft_radix2(elements);
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    return elapsed;
}
