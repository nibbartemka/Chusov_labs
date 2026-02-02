#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1u << 27)

#define MEASURE(func_name, func_call) \
    do { \
        double t0 = omp_get_wtime(); \
        double res = func_call; \
        double t1 = omp_get_wtime(); \
        printf("%-35s %8.4f с | результат: %.6f \n", \
            func_name, t1 - t0, res); \
    } while(0)


double avg(const double* v, size_t n) {
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        sum += v[i];
    }

    return sum / n;
}

double avg_omp_reduction(const double* v, size_t n) {
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += v[i];
    }
    
    return sum / n;
}

double avg_omp_parallel(const double* v, size_t n) {
    double sum = 0;

    #pragma omp parallel
    {
        size_t t = omp_get_thread_num();
        size_t T = omp_get_num_threads();

        size_t base = n / T;      // минимальный размер блока
        size_t remainder = n % T; // сколько потоков получат +1 элемент
        
        size_t start, size;
        
        if (t < remainder) {
            // Первые 'remainder' потоков получают на 1 элемент больше
            size = base + 1;
            start = t * size;
        } else {
            // Остальные потоки получают 'base' элементов
            size = base;
            start = remainder * (base + 1) + (t - remainder) * base;
        }

        double local_sum = 0;
        for (size_t i = start; i < start + size; i++) {
            local_sum += v[i];
        }

        #pragma omp critical
        sum += local_sum;
    }

    return sum / n;
}

double avg_omp_parallel_optimized(const double* v, size_t n) {
    unsigned P = omp_get_num_procs();

    double* r = malloc(P * sizeof(double));
    unsigned T;

    #pragma omp parallel shared(T)
    {
        size_t t = omp_get_thread_num();

        #pragma omp single
        {
            T = omp_get_num_threads();
        }

        double l_r = 0;

        for (size_t i = t; i < n; i += T) {
            l_r += v[i];
        }

        r[t] = l_r;
    }

    double total_r = 0;

    for (size_t i = 0; i < P; ++i) {
        total_r += r[i];
    }

    free(r);

    return total_r / n;
}

struct sum_t {
    double number;
    char padding[64 - sizeof(double)];
};

double avg_omp_with_cache_optimizing(const double* v, size_t n) {
    unsigned P = omp_get_num_procs();
    struct sum_t* r = calloc(P, sizeof(struct sum_t));

    unsigned T;

    #pragma omp parallel shared(T)
    {
        size_t t = omp_get_thread_num();

        #pragma omp single 
        {
            T = omp_get_num_threads();
        }

        double l_r = 0;

        for (size_t i = t; i < n; i+=T) {
            l_r += v[i];
        }

        r[t].number += l_r;
    }

    double total_r = 0;

    for (size_t i = 0; i < P; ++i)
    {
        total_r += r[i].number;
    }

    free(r);

    return total_r / n;
}

int main() {
    double* p = (double*)malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        p[i] = i;
    }

    MEASURE("Последовательная (avg)", avg(p, N));
    MEASURE("OpenMP reduction", avg_omp_reduction(p, N));
    MEASURE("OpenMP basic parallelism", avg_omp_parallel(p, N));
    MEASURE("OpenMP optimized parallelism", avg_omp_parallel_optimized(p, N));
    MEASURE("OpenMP + cache optimizing", avg_omp_with_cache_optimizing(p, N));

    free(p);

    return 0;
}