// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

#include "sleef.h"
#include "sleefredux.h"

#define STRINGIFY(x) #x
#define INLINE __attribute__((always_inline))
#define BILLION  1000000000LL

typedef struct timespec Timespec;
typedef uint64_t u64;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

static u64 elapsed_ns(Timespec start, Timespec stop) {
  return (u64)(stop.tv_sec - start.tv_sec) * BILLION + (u64)(stop.tv_nsec - start.tv_nsec);
}

#pragma clang diagnostic pop

// following two functions pulled from presum.c
static __m128 INLINE m128_scan(__m128 x) {
    // d    c   b  a
    // c    b   a  0 +
    // dc   cb  ba a
    // ba   a   0  0 +
    // dcba cba ba a
    x = _mm_add_ps(x, _mm_slli_si128(x, 4));
    // compiler chooses a movelh with zero
    x = _mm_add_ps(x, _mm_slli_si128(x, 8));
    return x;
}

static void INLINE scan_inplace_ss4(float* xs, size_t N, int dodiv, float div) {
    __builtin_assume(N >= 16);

    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_epi32(0);
    __m128 a, b, c, d, sa, sb, sc;
    __m128 vdiv = _mm_set1_epi32(div);
    for (size_t i = 0; i < N/4; i += 4) {
#define LOAD(x) (dodiv ? _mm_div_ps(x, vdiv) : x)
        a = _mm_add_ps(sum, m128_scan(LOAD(v[i])));

        b = m128_scan(LOAD(v[i + 1])); // hgfe gfe fe e
        c = m128_scan(LOAD(v[i + 2])); // lkji
        d = m128_scan(LOAD(v[i + 3])); // ponm
#undef LOAD

        sa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)); // sabcd
        sc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 3, 3, 3)); // ijkl

        b = _mm_add_ps(b, sa);
        d = _mm_add_ps(d, sc); // ijklmnop

        sb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)); // sabcdefgh
        c = _mm_add_ps(c, sb); // sabcdefghijkl
        d = _mm_add_ps(d, sb);

        sum = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 3, 3, 3));

        v[i + 0] = a;
        v[i + 1] = b;
        v[i + 2] = c;
        v[i + 3] = d;
    }
}

static void INLINE do_sum(float* xs, size_t N, int sumkind) {
    __builtin_assume(N >= 16);

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
    } else if (sumkind == 2) {
        scan_inplace_ss4(xs, N, 1, sum);
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

void softmax_sleefredux_presum_ss4(float* restrict src, float* restrict dst, size_t N) {
    do_exp(src, dst, N, 2, 0, 0.0);
    do_sum(dst, N, 2);
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

void softmax_sleefredux_presum_ss4_tempdiv(float* restrict src, float* restrict dst, size_t N, float temp) {
    do_exp(src, dst, N, 2, 1, temp);
    do_sum(dst, N, 2);
}

// -- tempmul
// these all ended up being the same as the compiler uses mul over div
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

#ifdef RUNTEST
#else

    Timespec start, stop;

    size_t rounds = 1000000;

    for (size_t N = 16; N <= 512; N *= 2) {
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

        BENCH(sleefredux_presum_ss4)
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

        BENCH(sleefredux_presum_ss4_tempdiv)

#undef BENCH

        free(xs);
    }

#endif
}
