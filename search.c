// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>

#include "random.h"

#define STRINGIFY(x) #x

typedef struct timespec Timespec;

#define BILLION  1000000000LL

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

static u64 elapsed_ns(Timespec start, Timespec stop) {
  return (u64)(stop.tv_sec - start.tv_sec) * BILLION + (u64)(stop.tv_nsec - start.tv_nsec);
}

#define INLINE __attribute__((always_inline))

// NOTE: in general, these return in the range [0, n] inclusive
// when searching a cdf, we expect v to be [0, 1) and xs[n-1] to be 1.0
// but with numerical error, xs[n-1] might happen to be 0.998 or something and v to
// be 0.999 in which case we will get n out, which is nonsensical for sampling
// so consumers need to clamp the max value at n-1 to be safe

size_t float_binary_search_cdf(float* xs, size_t n, float v) {
    float* cur = xs;
    while (n > 0) {
        if (v <= cur[n/2]) {
            n /= 2;
        } else {
            cur = &cur[n/2 + 1];
            n -= n/2 + 1;
        }
    }
    return cur - xs;
}

size_t float_linear_search_cdf(float* xs, size_t n, float v) {
    for (size_t i = 0; i < n; i++) {
        if (v <= xs[i]) {
            return i;
        }
    }
    return n;
}

// assumes no NaN
size_t float_xmm_search_cdf(float* xs, size_t n, float needle) {
    if (n < 16) return float_binary_search_cdf(xs, n, needle);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 needlev = _mm_set1_ps(needle);
    __m128 c;
    /*float* foo = aligned_alloc(16, sizeof(float) * 4);*/
    /*printf("needle is %.3f\n", needle);*/
    /*_mm_store_ps(foo, needlev);*/
    /*printf("needlev is %.3f  %.3f  %.3f  %.3f\n", needlev[0], needlev[1],needlev[2],needlev[3]);*/
    int m;
    for (size_t i = 0; i < n/4; i++) {
        // less than or equal, quiet
        c = _mm_cmp_ps(needlev, v[i], _CMP_LE_OQ);
        m = _mm_movemask_ps(c);
        if (m > 0) {
            // this gets the least significant set bit, so if more than one matches, we are okay
            int index = __builtin_ctz(m);
            return i*4 + index;
        }
    }
    return n;
}

void rng_init(PRNG32RomuQuad* rng, u8 seed) {
    assert(seed != 0);
    u32 x = seed;
    x = x | (x << 8) | (x << 16) | (x << 24);
    for (int i = 0; i < 4; i++) {
        rng->s[i] = x;
    }
    for (int i = 0; i < 10; i++) {
        (void)prng32_romu_quad(rng);
    }
}

float INLINE rng_float(PRNG32RomuQuad* rng) {
    return dist_uniformf(prng32_romu_quad(rng));
}

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.3f ", xs[i]);
    }
    printf("\n");
}

static void init_cdf(float* xs, size_t N, float v) {
    float sum = 0;
    for (size_t i = 0; i < N; i++) { xs[i] = v; }
    for (size_t i = 0; i < N; i++) { sum += xs[i]; }
    for (size_t i = 0; i < N; i++) { xs[i] /= sum; }
    for (size_t i = 1; i < N; i++) { xs[i] += xs[i-1]; }
}

static size_t round_up_size_t(size_t x, size_t N) {
    return ((x + (N - 1)) / N) * N;
}

int main(int argc, char** argv) {
    float v = 0.1;
    if (argc >= 2) { sscanf(argv[1], "%f", &v); }

    printf("init=%.2f\n", v);

    {
        for (size_t N = 8; N < 32; N++) {
            float* xs = aligned_alloc(16, round_up_size_t(sizeof(float)*N, 16));
            init_cdf(xs, N, 0.1);
            dump_array(xs, N);

            for (size_t i = 0; i < N; i++) {
                assert(i == float_binary_search_cdf(xs, N, xs[i]));
                assert(i == float_binary_search_cdf(xs, N, xs[i] - 0.01));
                assert(i == float_linear_search_cdf(xs, N, xs[i]));
                assert(i == float_linear_search_cdf(xs, N, xs[i] - 0.01));
                if (__builtin_popcount(N) == 1) {
                    /*printf("%ld %ld %.3f\n", i, float_xmm_search_cdf(xs, N, xs[i]), xs[i]);*/
                    assert(i == float_xmm_search_cdf(xs, N, xs[i]));
                    /*printf("%ld %ld %.3f\n", i, float_xmm_search_cdf(xs, N, xs[i] - 0.01), xs[i] - 0.01);*/
                    assert(i == float_xmm_search_cdf(xs, N, xs[i] - 0.01));
                }

                if (i == N-1) {
                    assert(i+1 == float_binary_search_cdf(xs, N, xs[i] + 0.01));
                    assert(i+1 == float_linear_search_cdf(xs, N, xs[i] + 0.01));
                } else {
                    assert(i+1 == float_binary_search_cdf(xs, N, xs[i] + 0.01));
                    assert(i+1 == float_linear_search_cdf(xs, N, xs[i] + 0.01));
                }
            }

            free(xs);
        }
    }

    Timespec start, stop;

    size_t rounds = 5000000;

    for (size_t N = 16; N <= 512; N *= 2) {
        float* xs = aligned_alloc(16, sizeof(float)*N);
        init_cdf(xs, N, 0.1);

        size_t check = 0;

        PRNG32RomuQuad rng;
        rng_init(&rng, 42);

        printf("N=%ld\n", N);

#define BENCH(name) \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds; i++) { \
            check += name(xs, N, rng_float(&rng)); \
        } \
        clock_ns(&stop); \
        printf("  %30s %.2f ns/call %.2f ms check=%lx\n", STRINGIFY(name), (double)elapsed_ns(start, stop) / (double)rounds, (double)elapsed_ns(start, stop) / 1000000, check);

        BENCH(float_binary_search_cdf);
        BENCH(float_linear_search_cdf);
        BENCH(float_xmm_search_cdf);


        free(xs);
    }

}
