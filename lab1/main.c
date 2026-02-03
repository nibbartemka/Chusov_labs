#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define N (1u << 27)

#define MEASURE(func_name, func_call) \
    do { \
        double t0 = omp_get_wtime(); \
        double res = func_call; \
        double t1 = omp_get_wtime(); \
        printf("%-35s %8.4f с | результат: %.6f \n", \
            func_name, t1 - t0, res); \
    } while(0)


#define SPEEDTEST(func, n, exp) \
    speedtest_avg_openmp(#func, func, n, exp)

void speedtest_avg_openmp(
    const char* name,
    double (*func)(const double*, size_t),
    size_t n,
    size_t exp_count
) {
    printf("======= SPEEDTEST OPEN MP AVG ==========\n");

    char filename[256];
    snprintf(filename, sizeof(filename), "%s.csv", name);

    FILE* output = fopen(filename, "w");
    if (!output) {
        printf("Error while opening file\n");
        return;
    }

    fprintf(output, "T,Time,Avg,Acceleration\n");

    double time_sum_1 = 0.0;
    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);

    for (int thread_num = 1; thread_num <= max_threads; thread_num++) {

        omp_set_num_threads(thread_num);

        double time_sum = 0.0;
        double t0 = omp_get_wtime();

        for (size_t exp = 0; exp < exp_count; exp++) {

            double* p = (double*)malloc(n * sizeof(double));
            if (!p) {
                printf("Memory allocation error\n");
                fclose(output);
                return;
            }

            for (size_t i = 0; i < n; i++) {
                p[i] = (double)i;
            }

            double t1 = omp_get_wtime();
            double result = func(p, n);
            double t2 = omp_get_wtime();

            time_sum += (t2 - t1) * 1000.0;

            free(p);
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
    // double* p = (double*)malloc(N * sizeof(double));

    // for (size_t i = 0; i < N; i++) {
    //     p[i] = i;
    // }

    // MEASURE("Последовательная (avg)", avg(p, N));
    // MEASURE("OpenMP reduction", avg_omp_reduction(p, N));
    // MEASURE("OpenMP basic parallelism", avg_omp_parallel(p, N));
    // MEASURE("OpenMP optimized parallelism", avg_omp_parallel_optimized(p, N));
    // MEASURE("OpenMP + cache optimizing", avg_omp_with_cache_optimizing(p, N));

    // free(p);

    SPEEDTEST(avg_omp_reduction, N, 5);
    SPEEDTEST(avg_omp_parallel, N, 5);
    SPEEDTEST(avg_omp_parallel_optimized, N, 5);
    SPEEDTEST(avg_omp_with_cache_optimizing, N, 5);

    return 0;
}