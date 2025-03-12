#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

#include "sleef.h"
#include "sleefredux.h"

#define STRINGIFY(x) #x

typedef struct timespec Timespec;


#define BILLION  1000000000LL

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

typedef uint64_t u64;

static u64 elapsed_ns(Timespec start, Timespec stop) {
  u64 acpre = (u64)(stop.tv_sec - start.tv_sec) * BILLION
               + (u64)(stop.tv_nsec - start.tv_nsec);
  return acpre;
}

#define INLINE __attribute__((always_inline))

static void INLINE do_sum(float* xs, size_t N, int sumkind) {
    __builtin_assume(N >= 8);

    float sum = 0;
    for (size_t i = 0; i < N; i++) {
        sum += xs[i];
    }
    if (sumkind == 0) {
        for (size_t i = 0; i < N; i++) {
            xs[i] /= sum;
        }
    } else if (sumkind == 1) {
        xs[0] /= sum;
        for (size_t i = 1; i < N; i++) {
            xs[i] = xs[i - 1] + xs[i] / sum;
        }
    }
}

static void INLINE do_exp(float* restrict src, float* restrict dst, size_t N, int kind, int tempkind, float temp) {
    __builtin_assume(N >= 8);

    float tempinv = 1 / temp;
    __m256 tempv = _mm256_set1_ps(temp);
    __m256 tempinvv = _mm256_set1_ps(1 / temp);
    if (kind == 0) {
        if (tempkind == 0) {
            for (size_t i = 0; i < N; i++) { dst[i] = expf(src[i]); }
        } else if (tempkind == 1) {
            for (size_t i = 0; i < N; i++) { dst[i] = expf(src[i] / temp); }
        } else if (tempkind == 2) {
            for (size_t i = 0; i < N; i++) { dst[i] = expf(src[i] * tempinv); }
        }
    } else if (kind == 1) {
        __m256* restrict srcv = __builtin_assume_aligned(src, 32);
        __m256* restrict dstv = __builtin_assume_aligned(dst, 32);
        if (tempkind == 0) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_finz_expf8_u10avx2(srcv[i]); }
        } else if (tempkind == 1) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_finz_expf8_u10avx2(_mm256_div_ps(srcv[i], tempv)); }
        } else if (tempkind == 2) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_finz_expf8_u10avx2(_mm256_mul_ps(srcv[i], tempinvv)); }
        }
    } else if (kind == 2) {
        __m256* restrict srcv = __builtin_assume_aligned(src, 32);
        __m256* restrict dstv = __builtin_assume_aligned(dst, 32);
        if (tempkind == 0) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_redux_finz_expf8_u10avx2(srcv[i]); }
        } else if (tempkind == 1) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_redux_finz_expf8_u10avx2(_mm256_div_ps(srcv[i], tempv)); }
        } else if (tempkind == 2) {
            for (size_t i = 0; i < N/8; i++) { dstv[i] = Sleef_redux_finz_expf8_u10avx2(_mm256_mul_ps(srcv[i], tempinvv)); }
        }
    }
}

// -- notemp

void softmax_math_sum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 0, 0, 0.0);
    do_sum(dst, N, 0);
}

void softmax_math_presum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 0, 0, 0.0);
    do_sum(dst, N, 1);
}

void softmax_sleef_sum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 1, 0, 0.0);
    do_sum(dst, N, 0);
}

void softmax_sleef_presum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 1, 0, 0.0);
    do_sum(dst, N, 1);
}

void softmax_sleefredux_sum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 2, 0, 0.0);
    do_sum(dst, N, 0);
}

void softmax_sleefredux_presum(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 2, 0, 0.0);
    do_sum(dst, N, 1);
}

// -- tempdiv

void softmax_math_sum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 0, 1, temp);
    do_sum(dst, N, 0);
}

void softmax_math_presum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 0, 1, temp);
    do_sum(dst, N, 1);
}

void softmax_sleef_sum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 1, 1, temp);
    do_sum(dst, N, 0);
}

void softmax_sleef_presum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 1, 1, temp);
    do_sum(dst, N, 1);
}

void softmax_sleefredux_sum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 2, 1, temp);
    do_sum(dst, N, 0);
}

void softmax_sleefredux_presum_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 2, 1, temp);
    do_sum(dst, N, 1);
}

// -- tempmul
void softmax_math_sum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 0, 2, temp);
    do_sum(dst, N, 0);
}

void softmax_math_presum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 0, 2, temp);
    do_sum(dst, N, 1);
}

void softmax_sleef_sum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 1, 2, temp);
    do_sum(dst, N, 0);
}

void softmax_sleef_presum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 1, 2, temp);
    do_sum(dst, N, 1);
}

void softmax_sleefredux_sum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 2, 2, temp);
    do_sum(dst, N, 0);
}

void softmax_sleefredux_presum_tempmul(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 2, 2, temp);
    do_sum(dst, N, 1);
}

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.3f ", xs[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    float v = 0.1;
    float temp = 0.1;
    if (argc >= 2) { sscanf(argv[1], "%f", &v); }
    if (argc >= 3) { sscanf(argv[1], "%f", &temp); }

    printf("init=%.2f temp=%.2f\n", v, temp);

    Timespec start, stop;

    size_t rounds = 1000000;

    for (size_t N = 8; N <= 512; N *= 2) {
        float* xs = aligned_alloc(32, sizeof(float)*N*2);
        float* dst = xs + N;
        for (size_t i = 0; i < N; i++) {
            xs[i] = v;
        }

        printf("N=%ld\n", N);

#define BENCH(name) \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds; i++) { \
            softmax_##name(xs, dst, N); \
        } \
        clock_ns(&stop); \
        printf("  %30s %.2f ns/el %.2f ms\n", STRINGIFY(name), (double)elapsed_ns(start, stop) / (double)rounds / (double)N, (double)elapsed_ns(start, stop) / 1000000);

        BENCH(math_sum)
        BENCH(sleef_sum)
        BENCH(sleefredux_sum)

        BENCH(math_presum)
        BENCH(sleef_presum)
        BENCH(sleefredux_presum)
#undef BENCH

#define BENCH(name) \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds; i++) { \
            softmax_##name(xs, dst, N, temp); \
        } \
        clock_ns(&stop); \
        printf("  %30s %.2f ns/el %.2f ms\n", STRINGIFY(name), (double)elapsed_ns(start, stop) / (double)rounds / (double)N, (double)elapsed_ns(start, stop) / 1000000);

        // inspecting the asm, compiler changes the division to a multiplication anyways
        BENCH(math_sum_tempdiv)
        /*BENCH(math_sum_tempmul)*/

        BENCH(sleef_sum_tempdiv)
        /*BENCH(sleef_sum_tempmul)*/

        BENCH(sleefredux_sum_tempdiv)
        /*BENCH(sleefredux_sum_tempmul)*/

        BENCH(math_presum_tempdiv)
        /*BENCH(math_presum_tempmul)*/

        BENCH(sleef_presum_tempdiv)
        /*BENCH(sleef_presum_tempmul)*/

        BENCH(sleefredux_presum_tempdiv)
        /*BENCH(sleefredux_presum_tempmul)*/

#undef BENCH

        free(xs);
    }
}
