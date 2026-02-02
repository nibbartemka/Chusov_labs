#include <complex>
#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <algorithm>
#include <chrono>


void dft_generic(const std::complex<double>* input, std::complex<double>* output, std::size_t n, std::complex<double> w) {
    std::complex<double> Wk = 1.0;

    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> sum = 0.0;

        std::complex<double> p = 1.0;
        for (std::size_t t = 0; t < n; ++t) {
            sum += input[t] * p;
            p *= Wk;
        }

        output[k] = sum;
        Wk *= w;
    }
}

void dft(const std::complex<double>* time, std::complex<double>* spectrum, size_t n) {
    const double pi = std::numbers::pi_v<double>;
    const std::complex<double> w = std::polar(1.0, -2.0 * pi / static_cast<double>(n));
    dft_generic(time, spectrum, n, w);
};

void idft(const std::complex<double>* spectrum, std::complex<double>* time, size_t n) {
    const double pi = std::numbers::pi_v<double>;
    const std::complex<double> w = std::polar(1.0, +2.0 * pi / static_cast<double>(n));
    dft_generic(spectrum, time, n, w);

    const double inv_n = 1.0 / static_cast<double>(n);
    for (std::size_t t = 0; t < n; ++t) time[t] *= inv_n;
};

void test_dft_singleprocessing(){
    const std::size_t N = 4;
    std::vector<std::complex<double>> x = {3.0, 4.0, 0.0, 0.0};
    std::vector<std::complex<double>> y = {4.0, 3.0, 0.0, 0.0};

    std::vector<std::complex<double>> X(N), Y(N), Z(N);
    dft(x.data(), X.data(), N);
    dft(y.data(), Y.data(), N);

    for (std::size_t k = 0; k < N; ++k)
        Z[k] = X[k] * Y[k];

    std::vector<std::complex<double>> z_fft(N);
    idft(Z.data(), z_fft.data(), N);

    std::vector<std::complex<double>> z_circ(N, 0.0);
    for (std::size_t n = 0; n < N; ++n) {
        std::complex<double> s = 0.0;
        for (std::size_t m = 0; m < N; ++m) {
            std::size_t idx = (n + N - m) % N;
            s += x[m] * y[idx];
        }
        z_circ[n] = s;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "x: "; for (auto v : x) std::cout << v << " ";
    std::cout << "\ny: "; for (auto v : y) std::cout << v << " ";

    std::cout << "\n\nКруговая свёртка через теорему (IDFT(DFT(x) * DFT(y))):\n";
    for (std::size_t n = 0; n < N; ++n)
        std::cout << "z_fft[" << n << "] = " << z_fft[n] << "\n";

    std::cout << "\nКруговая свёртка (ручной подсчёт):\n";
    for (std::size_t n = 0; n < N; ++n)
        std::cout << "z_circ[" << n << "] = " << z_circ[n] << "\n";

    double max_err = 0.0;
    for (std::size_t n = 0; n < N; ++n)
        max_err = std::max(max_err, std::abs(z_fft[n] - z_circ[n]));

    std::cout << "\nМакс. расхождение: " << max_err << "\n";

    std::cout << "\nОжидаем (теоретически): [12, 25, 12, 0]\n";
}

std::chrono::duration<double> test_dft_time(std::vector<std::complex<double>> elements) {
    auto time_start = std::chrono::steady_clock::now(); 
    std::vector<std::complex<double>> result(elements.size());
    dft(elements.data(), result.data(), elements.size());
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;
    return elapsed;
}
