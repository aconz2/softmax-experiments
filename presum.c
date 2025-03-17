// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <assert.h>
#include <time.h>
#include <math.h>

typedef struct timespec Timespec;

#define BILLION  1000000000LL

#define STRINGIFY(x) #x
#define INLINE __attribute__((always_inline))

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

__m128 m128_scan(__m128 x) {
    // d    c   b  a
    // c    b   a  0 +
    // dc   cb  ba a
    // ba   a   0  0 +
    // dcba cba ba a
    x = _mm_add_ps(x, _mm_slli_si128(x, 4));
    // or a movelh with zero
    x = _mm_add_ps(x, _mm_slli_si128(x, 8));
    return x;
}

void scan_simple(float* xs, size_t n) {
    for (size_t i = 1; i < n; i++) {
        xs[i] += xs[i - 1];
    }
}

void INLINE scan_inplace_(float* xs, size_t n, float sum_in) {
    assert(n % 4 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_ps(sum_in);
    __m128 x;
    // this gets unrolled by 8
    for (size_t i = 0; i < n/4; i++) {
        x = _mm_add_ps(sum, m128_scan(v[i]));
        v[i] = x;
        sum = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

void scan_inplace(float* xs, size_t n) {
    scan_inplace_(xs, n, 0);
}

void scan_inplace_ss2_(float* xs, size_t n, float sum_in) {
    assert(n % 8 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_ps(sum_in);
    __m128 a, b, sa;
    for (size_t i = 0; i < n/4; i += 2) {
        a = _mm_add_ps(sum, m128_scan(v[i])); // sdcba scba sba sa

        b = m128_scan(v[i + 1]); // hgfe gfe fe e
        sa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3));

        b = _mm_add_ps(b, sa);

        sum = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3));

        v[i + 0] = a;
        v[i + 1] = b;
    }
}

void scan_inplace_ss2(float* xs, size_t n) {
    scan_inplace_ss2_(xs, n, 0);
}

void scan_inplace_ss4_(float* xs, size_t n, float sum_in) {
    assert(n % 16 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_ps(sum_in);
    __m128 a, b, c, d, sa, sb, sc;
    for (size_t i = 0; i < n/4; i += 4) {
        a = _mm_add_ps(sum, m128_scan(v[i]));

        b = m128_scan(v[i + 1]); // hgfe gfe fe e
        c = m128_scan(v[i + 2]); // lkji
        d = m128_scan(v[i + 3]); // ponm

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

void scan_inplace_ss4(float* xs, size_t n) {
    scan_inplace_ss4_(xs, n, 0);
}

void scan_unaligned_simple(float* xs, size_t n) {
    if (n < 4) {
        scan_simple(xs, n);
        return;
    }

    int valid = 0;

    while (((uintptr_t)xs) & 0b1111) {
        if (valid) {
            *xs += *(xs - 1);
        }
        valid = 1;
        xs += 1;
        n -= 1;
    }

    size_t n_aligned = (n / 16) * 16;
    scan_inplace_ss4_(xs, n_aligned, valid ? xs[-1] : 0);
    xs += n_aligned;
    n -= n_aligned;
    valid = valid || (n_aligned > 0);
    /*printf("n_rem=%d\n", n);*/
    if (valid) {
        for (ssize_t i = 0; i < n; i++) {
            xs[i] += xs[i-1];
        }
    } else {
        for (ssize_t i = 1; i < n; i++) {
            xs[i] += xs[i-1];
        }
    }
}

void scan_unaligned(float* xs, size_t n) {
    if (n < 4) {
        scan_simple(xs, n);
        return;
    }

    int valid = 0;

#ifndef NDEBUG
    size_t head = 0;
#endif

    while (((uintptr_t)xs) & 0b1111) {
        if (valid) {
            *xs += *(xs-1);
        }
        xs += 1;
        n -= 1;
#ifndef NDEBUG
        head += 1;
#endif
        valid = 1;
    }

#ifndef NDEBUG
    /*printf(" n_head=%ld\n", head);*/
#endif

    { // 16
        size_t n_aligned = (n / 16) * 16;
        /*printf(" n_aligned16=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_ss4_(xs, n_aligned, valid ? xs[-1] : 0);
        xs += n_aligned;
        n -= n_aligned;
        valid = valid || (n_aligned > 0);
    }
    { // 8
        size_t n_aligned = (n / 8) * 8;
        /*printf(" n_aligned8=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_ss2_(xs, n_aligned, valid ? xs[-1] : 0);
        xs += n_aligned;
        n -= n_aligned;
        valid = valid || (n_aligned > 0);
    }
    { // 4
        size_t n_aligned = (n / 4) * 4;
        /*printf(" n_aligned4=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_(xs, n_aligned, valid ? xs[-1] : 0);
        xs += n_aligned;
        n -= n_aligned;
    }
    assert(n <= 3);
    /*printf(" n_rem=%ld\n", n);*/
    for (ssize_t i = 0; i < n; i++) {
        xs[i] += xs[i-1];
    }
}

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", xs[i]);
    }
    printf("\n");
}

static void init_norm(float* xs, size_t N, float v) {
    float sum = 0;
    for (size_t i = 0; i < N; i++) { xs[i] = v; }
    for (size_t i = 0; i < N; i++) { sum += xs[i]; }
    for (size_t i = 0; i < N; i++) { xs[i] /= sum; }
}

float total_diff(float* ref, float* xs, size_t N) {
    float sum = 0;
    for (size_t i = 0; i < N; i++) {
        sum += fabs(ref[i] - xs[i]);
    }
    return sum;
}

int main() {

#ifdef RUNTEST
    // this is a fudged number by empirical testing of what the tests have done
    const double MAX_AVG_DIFF = 1e-06;
    {
        size_t N = 32;
        float diff = 0;
        float* xs = aligned_alloc(16, sizeof(float)*N);
        float* ref = aligned_alloc(16, sizeof(float)*N);
        // for testing we want to concentrate on the case of numbers in [0, 1)
        init_norm(xs, N, 0.1);
        init_norm(ref, N, 0.1);
        dump_array(xs, N);

        scan_simple(ref, N);
        dump_array(ref, N);

#define TEST(name) \
        name(xs, N); \
        dump_array(xs, N); \
        diff = total_diff(ref, xs, N);
        printf("%s total diff %f avg diff %e\n", STRINGIFY(name), diff, (double)diff / N); \
        assert((double)diff / N <= MAX_AVG_DIFF); \
        init_norm(xs, N, 0.1);

        TEST(scan_simple);
        TEST(scan_inplace);
        TEST(scan_inplace_ss2);
        TEST(scan_inplace_ss4);
        TEST(scan_unaligned_simple);
#undef TEST

        free(xs);
        free(ref);
    }
    {
        float diff = 0;
        size_t N = 48;
        float* xs = aligned_alloc(16, sizeof(float)*N);
        float* ref = aligned_alloc(16, sizeof(float)*N);

        // to test all unalignment cases
        // remove i elements from the front and j elements from the back
                /*dump_array(ref + i, n); \*/
                /*dump_array(xs + i, n); \*/
#define TESTU(name) \
        for (size_t N = 8; N <= 48; N += 8) { \
            for (size_t i = 0; i < 4; i++) { \
                for (size_t j = 0; j < 4; j++) { \
                    size_t n = N - i - j; \
                    init_norm(xs + i, n, 0.1); \
                    init_norm(ref + i, n, 0.1); \
                    scan_simple(ref + i, n); \
                    name(xs + i, n); \
                    diff = total_diff(ref + i, xs + i, n); \
                    printf("unaligned %s n=%ld i=%ld j=%ld total diff %f avg diff %e\n", STRINGIFY(name), n, i, j, diff, (double)diff / n); \
                    assert((double)diff / n <= MAX_AVG_DIFF); \
                } \
            } \
        }

        TESTU(scan_unaligned_simple);
        TESTU(scan_unaligned);

#undef TESTU
        free(xs);
        free(ref);
    }
#else

    size_t rounds = 100000000;

    Timespec start, stop;

    for (size_t N = 16; N <= 2048; N *= 2) {
        float* xs = aligned_alloc(16, sizeof(float)*N);
        for (size_t i = 0; i < N; i++) { xs[i] = 0.1 * i; }
        printf("N=%ld\n", N);

#define BENCH(name) \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds / N; i++) { \
            name(xs, N); \
            for (size_t i = 0; i < N; i++) { xs[i] = 0.1 * i; } \
        } \
        clock_ns(&stop); \
        printf("  %20s %.2f ns/el %.2f ms\n", #name, (double)elapsed_ns(start, stop) / (double)(rounds / N) / (double)N, (double)elapsed_ns(start, stop) / 1000000);

        BENCH(scan_simple);
        BENCH(scan_inplace);
        BENCH(scan_inplace_ss2);
        BENCH(scan_inplace_ss4);
        BENCH(scan_unaligned);

        free(xs);

#undef BENCH

    }
#endif
}
