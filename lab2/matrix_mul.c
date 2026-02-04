#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define R_DIM 512   // Количество строк матрицы A и D
#define N_DIM 512   // Внутренняя размерность (столбцы A, строки B)
#define C_DIM 512   // Количество столбцов матрицы B и D

#define VEC_SIZE (sizeof(__m256d) / sizeof(double))

#define SPEEDTEST(func, exp) \
    speedtest_mul_matrix(#func, func, exp)

void speedtest_mul_matrix(
    const char* name,
    void (*func)(double* D, const double* A, const double* B, 
                      size_t n, size_t R, size_t C),
    size_t exp_count
) {
    printf("======= SPEEDTEST MATRIX MUL ==========\n");


    char filename[256];
    snprintf(filename, sizeof(filename), "%s.csv", name);

    FILE* output = fopen("matrix_output.csv", "a");
    // FILE* output = fopen(filename, "w");
    
    if (!output) {
        printf("Error while opening file\n");
        return;
    }

    double time_sum_1 = 0.0;

    double time_sum = 0.0;
    double t0 = omp_get_wtime();

    for (size_t exp = 0; exp < exp_count; exp++) {

        size_t R = R_DIM, n = N_DIM, C = C_DIM;
        size_t size_A = R * n;
        size_t size_B = n * C;
        size_t size_D = R * C;

        double *A = (double*)aligned_alloc(32, size_A * sizeof(double));
        double *B = (double*)aligned_alloc(32, size_B * sizeof(double));
        double *D1 = (double*)aligned_alloc(32, size_D * sizeof(double));
        double *D2 = (double*)aligned_alloc(32, size_D * sizeof(double));

        for (size_t k = 0; k < n; k++) {
            for (size_t i = 0; i < R; i++) {
                A[k * R + i] = i;
            }
        }

        for (size_t j = 0; j < C; j++) {
            for (size_t k = 0; k < n; k++) {
                B[j * n + k] = k;
            }
        }

        double t1 = omp_get_wtime();
        func(D1, A, B, n, R, C);
        double t2 = omp_get_wtime();

        time_sum += (t2 - t1) * 1000.0;

        free(A);
        free(B);
        free(D1);
        free(D2);
    }

    double total_time = (omp_get_wtime() - t0) * 1000.0;

    double avg = time_sum / exp_count;

    printf(
        "MUL: Method = %s\t| total experiment time: %.2f ms\t| avg time = %.2f ms\n",
        name, total_time, avg
    );

    fprintf(
        output, "%s,%.2f,%.2f\n",
        name, total_time, avg
    );
    

    fclose(output);
}


void matrix_mul(double* D, const double* A, const double* B, 
               size_t n, size_t R, size_t C)
{
    for (size_t i = 0; i < R; i++)
        for (size_t j = 0; j < C; j++) {
            double accum = 0;
            
            for (size_t k = 0; k < n; k++)
                accum += A[i * n + k] * B[k * C + j];
            D[i * C + j] = accum;
        }
}


void matrix_mul_colmajor(double* D, const double* A, const double* B, 
                        size_t n, size_t R, size_t C)
{
    for (size_t i = 0; i < R; i++) {
        for (size_t j = 0; j < C; j++) {
            double accum = 0.0;
            for (size_t k = 0; k < n; k++) {
                // A[i][k] в column-major: A[k * R + i]
                // B[k][j] в column-major: B[j * n + k]
                accum += A[k * R + i] * B[j * n + k];
            }
            // D[i][j] в column-major: D[j * R + i]
            D[j * R + i] = accum;
        }
    }
}


// Используем AVX для column-major матриц 
void matrix_mul_avx256(double* D, const double* A, const double* B, 
                      size_t n, size_t R, size_t C)
{
    assert(R % VEC_SIZE == 0);

    for (size_t i = 0; i < R / VEC_SIZE; i++) {
        for (size_t j = 0; j < C; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < n; k++) {
                __m256d x = _mm256_load_pd(&A[k * R + i * VEC_SIZE]);
                __m256d y = _mm256_set1_pd(B[j * n + k]);
                sum = _mm256_fmadd_pd(x, y, sum);
            }
            _mm256_storeu_pd(&D[j * R + i * VEC_SIZE], sum);
        }
    }
}

void create_file() {
    FILE* output = fopen("matrix_output.csv", "w");
    if (!output) {
        printf("Error while opening file\n");
        return;
    }
    fprintf(output, "Method,Time,Avg\n");
    fclose(output);
}

int main() {
    size_t test_count = 15;
    create_file();
    SPEEDTEST(matrix_mul, test_count);
    SPEEDTEST(matrix_mul_colmajor, test_count);
    SPEEDTEST(matrix_mul_avx256, test_count);    

    // size_t R = R_DIM, n = N_DIM, C = C_DIM;
    // size_t size_A = R * n;
    // size_t size_B = n * C;
    // size_t size_D = R * C;

    // // Выделяем выровненную память (32 байта для AVX)
    // double *A = (double*)aligned_alloc(32, size_A * sizeof(double));
    // double *B = (double*)aligned_alloc(32, size_B * sizeof(double));
    // double *D1 = (double*)aligned_alloc(32, size_D * sizeof(double));
    // double *D2 = (double*)aligned_alloc(32, size_D * sizeof(double));

    // // === Инициализация матриц НАПРЯМУЮ в COLUMN-MAJOR формате ===
    // // Матрица A (R строк × n столбцов): элемент (i, k) → A[k * R + i]
    // for (size_t k = 0; k < n; k++) {
    //     for (size_t i = 0; i < R; i++) {
    //         A[k * R + i] = i;  // Пример заполнения
    //     }
    // }

    // // Матрица B (n строк × C столбцов): элемент (k, j) → B[j * n + k]
    // for (size_t j = 0; j < C; j++) {
    //     for (size_t k = 0; k < n; k++) {
    //         B[j * n + k] = k;  // Пример заполнения
    //     }
    // }

    // // === Наивное умножение (column-major) ===
    // clock_t start = clock();
    // matrix_mul_colmajor(D1, A, B, n, R, C);
    // clock_t end = clock();
    // double time_naive = (double)(end - start) / CLOCKS_PER_SEC;

    // // === AVX256 умножение (column-major) ===
    // start = clock();
    // matrix_mul_avx256(D2, A, B, n, R, C);
    // end = clock();
    // double time_avx = (double)(end - start) / CLOCKS_PER_SEC;

    // // === Сравнение результатов ===
    // double max_abs_err = 0.0;
    // double max_rel_err = 0.0;
    // size_t err_pos = 0;

    // for (size_t idx = 0; idx < size_D; idx++) {
    //     double ref = D1[idx];
    //     double val = D2[idx];
    //     double abs_err = fabs(ref - val);
    //     double rel_err = (fabs(ref) > 1e-12) ? abs_err / fabs(ref) : abs_err;

    //     if (abs_err > max_abs_err) {
    //         max_abs_err = abs_err;
    //         err_pos = idx;
    //     }
    //     if (rel_err > max_rel_err) {
    //         max_rel_err = rel_err;
    //     }
    // }

    // // === Вывод результатов ===
    // printf("Размеры матриц: A[%zux%zu] * B[%zux%zu] = D[%zux%zu]\n", 
    //        R, n, n, C, R, C);

    // printf("=== Производительность ===\n");
    // printf("Наивная реализация: %.4f с\n", time_naive);
    // printf("AVX256 реализация:  %.4f с\n", time_avx);

    // printf("\n=== Корректность ===\n");
    // printf("Макс. абсолютная ошибка: %e\n", max_abs_err);
    // printf("Макс. относительная ошибка: %e\n", max_rel_err);
    // printf("Позиция ошибки: [%zu][%zu] (строка, столбец)\n", 
    //        err_pos % R, err_pos / R);

    // if (max_rel_err < 1e-12) {
    //     printf("Результаты совпадают (в пределах точности double)\n");
    // } else {
    //     printf("Результаты НЕ совпадают! Относительная ошибка = %e\n", max_rel_err);
    // }

    // free(A);
    // free(B);
    // free(D1);
    // free(D2);

    return 0;
}