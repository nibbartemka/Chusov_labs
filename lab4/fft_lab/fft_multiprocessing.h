#include <chrono>
#include <complex>
#include <vector>

void fft_radix2_openmp(std::complex<double>* x, size_t n, int maxDepth);
void ifft_radix2_openmp(std::complex<double>* X, size_t n, int maxDepth);