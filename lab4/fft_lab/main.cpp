#include "fft.h"
#include "dft.h"
#include "fft_multiprocessing.h"
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <numbers>
#include <random>

constexpr size_t MAX_TASK_DEPTH = 100;


void randomize_vector(std::vector<std::complex<double>>& v) {
    std::uniform_real_distribution<double> unif(0, 100000);
    static std::random_device rd;
    std::default_random_engine re(rd());
    for (auto & i : v)
    {
        i = unif(re);
    }
};

void speedtest_fft_radix2_tasked(size_t n, size_t exp_count) {
    printf("======= SPEEDTEST FFT ==========\n");

    FILE* output = fopen("fft_output.csv", "w");
    if (!output) {
        printf("Error while opening file\n");
        return;
    }

    fprintf(output, "T,Time,Avg,Acceleration\n");

    double time_sum_1 = 0.0;
    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);

    std::vector<std::complex<double>> original(n);
    randomize_vector(original);

    std::vector<std::complex<double>> original_copy = original;


    for (int thread_num = 1; thread_num <= max_threads; thread_num++) {

        omp_set_num_threads(thread_num);
        
        original = original_copy;

        double time_sum = 0.0;
        double t0 = omp_get_wtime();

        for (size_t exp = 0; exp < exp_count; exp++) {

            std::vector<std::complex<double>> spectrum = original;
            std::vector<std::complex<double>> restored(n);


            double t1 = omp_get_wtime();
            fft_radix2_openmp(spectrum.data(), n, MAX_TASK_DEPTH);
            ifft_radix2_openmp(spectrum.data(), n, MAX_TASK_DEPTH);
            double t2 = omp_get_wtime();

            time_sum += (t2 - t1) * 1000.0;
        }

        double total_time = (omp_get_wtime() - t0) * 1000.0;

        if (thread_num == 1) {
            time_sum_1 = time_sum;
        }

        double avg = time_sum / exp_count;
        double acceleration = (time_sum_1 / exp_count) / avg;

        printf(
            "AVG: T = %d\t| total experiment time: %.2f ms\t| avg time = %.2f ms\t| acceleration = %.2f\n",
            thread_num, total_time, avg, acceleration
        );

        fprintf(
            output, "%d,%.2f,%.2f,%.2f\n",
            thread_num, total_time, avg, acceleration
        );
    }

    fclose(output);
}



int main () {
    speedtest_fft_radix2_tasked(1 << 18, 5);
    return 0;
}