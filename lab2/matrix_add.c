#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>

#define NUM_RUNS 100


void matrix_add(double* D, const double* A, const double* B, 
                unsigned C, unsigned R) {
    for (size_t c = 0; c < C; ++c) {
        for (size_t r = 0; r < R; r++) {
            D[c*R + r] = A[c*R + r] + B[c*R + r];
        }
    }
}

void matrix_add_avx(double* D, const double* A, const double* B, 
                    unsigned C, unsigned R) {
    assert(R % sizeof(__m256) == 0);

    for (size_t c = 0; c < C; ++c) {
        for (size_t r = 0; r < R; r += sizeof(__m256d)/sizeof(double)) {
            __m256d x = _mm256_loadu_pd(&A[c*R + r]);
            __m256d y = _mm256_loadu_pd(&B[c*R + r]);
            _mm256_storeu_pd(&D[c*R + r], _mm256_add_pd(x, y));
        }
    }
}

// Проверка корректности
int verify_results(const double* D1, const double* D2, unsigned C, unsigned R, double epsilon) {
    for (size_t c = 0; c < C; ++c) {
        for (size_t r = 0; r < R; r++) {
            double diff = D1[c*R + r] - D2[c*R + r];
            if (diff < -epsilon || diff > epsilon) {
                printf("Ошибка на позиции [%zu][%zu]: %.10f != %.10f\n", 
                       c, r, D1[c*R + r], D2[c*R + r]);
                return 0;
            }
        }
    }
    return 1;
}

// Измерение времени
double measure_time(void (*func)(double*, const double*, const double*, unsigned, unsigned),
                    double* D, const double* A, const double* B,
                    unsigned C, unsigned R, int runs) {
    double total_time = 0;
    
    for (int i = 0; i < runs; i++) {
        clock_t start = clock();
        func(D, A, B, C, R);
        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }
    
    return total_time / runs;
}

int main() {
    unsigned C = 512; 
    unsigned R = 512; 

    size_t total_elements = C * R;

    double* A = (double*)aligned_alloc(32, total_elements * sizeof(double));
    double* B = (double*)aligned_alloc(32, total_elements * sizeof(double));
    double* D1 = (double*)aligned_alloc(32, total_elements * sizeof(double));
    double* D2 = (double*)aligned_alloc(32, total_elements * sizeof(double));

    printf("\nТестирование производительности (%d запусков):\n", NUM_RUNS);

    double time_simple = measure_time(matrix_add, D1, A, B, C, R, NUM_RUNS);
    double time_avx = measure_time(matrix_add_avx, D2, A, B, C, R, NUM_RUNS);

    printf("\nПроверка корректности...\n");
    int valid1 = verify_results(D1, D2, C, R, 1e-10);

    if (valid1) {
        printf("Результаты совпадают\n");
    } else {
        printf("Результаты не совпадают!\n");
    }

    printf("Простой алгоритм:    %.6f сек\n", time_simple);
    printf("AVX алгоритм:        %.6f сек\n", time_avx);

    free(A);
    free(B);
    free(D1);
    free(D2);
}