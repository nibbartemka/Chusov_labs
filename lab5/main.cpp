#include "fft.h"
#include "dft.h"
#include "fft_multiprocessing.h"
#include <iostream>

int main () {
    std::vector<std::complex<double>> v(1<<15);

    for (size_t i = 0; i < v.size() / 2; ++i) {
        v[i] = v.rbegin()[i] = i;
    }
    auto elasped_dft = test_dft_time(v);
    auto elasped_fft = test_fft_time(v);
    auto elasped_ftt_multiprocessing = test_fft_multiprocessing_time(v);
    std::cout << elasped_dft.count() << " " << elasped_fft.count() << " " << elasped_ftt_multiprocessing.count() << "\n";
    return 0;
}