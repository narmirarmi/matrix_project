#ifndef TIMING_H
#define TIMING_H

#include <omp.h>
#include <time.h>
#include <stdio.h>

typedef struct {
    clock_t cpu_start;
    double wall_start;
} TimingInfo;

#define TICK(X) TimingInfo X; \
X.cpu_start = clock(); \
X.wall_start = omp_get_wtime()

#define TOCK(X) do { \
clock_t cpu_end = clock(); \
double wall_end = omp_get_wtime(); \
printf("CPU Time: %f seconds\n", (double)(cpu_end - X.cpu_start) / CLOCKS_PER_SEC); \
printf("Wall Clock Time: %f seconds\n", wall_end - X.wall_start); \
} while(0)

#endif // TIMING_H