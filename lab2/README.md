# Лабораторная 2
Применение регистров AVX для ускорения выполнения операций над матрицами

Сложение:
```
gcc -march=native matrix_add.c -o matrix_add
./matrix_add
```

Умножение:
```
gcc -march=native -fopenmp matrix_mul.c -o matrix_mul
./matrix_mul
```

