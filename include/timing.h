#ifndef TIMING_H
#define TIMING_H

#include <omp.h>
#include <time.h>
#include <stdio.h>

typedef struct {
    clock_t cpu_start;
    double wall_start;
    double cpu_time;
    double wall_time;
} TimingInfo;

#define TICK(X) TimingInfo X; \
X.cpu_start = clock(); \
X.wall_start = omp_get_wtime()

#define TOCK(X) do { \
clock_t cpu_end = clock(); \
double wall_end = omp_get_wtime(); \
X.cpu_time = (double)(cpu_end - X.cpu_start) / CLOCKS_PER_SEC; \
X.wall_time = wall_end - X.wall_start; \
printf("CPU Time: %f seconds\n", X.cpu_time); \
printf("Wall Clock Time: %f seconds\n", X.wall_time); \
} while(0)

#endif // TIMING_H