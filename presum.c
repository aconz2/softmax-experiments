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
#define NOINLINE __attribute__((noinline))

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

void scan_simple_(float* xs, size_t n, float sum) {
    if (n == 0) return;
    xs[0] += sum;
    for (size_t i = 1; i < n; i++) {
        xs[i] += xs[i - 1];
    }
}

void NOINLINE scan_simple(float* xs, size_t n) {
    scan_simple_(xs, n, 0);
}

__m128 INLINE scan_inplace_(float* xs, size_t n, __m128 sum) {
    assert(n % 4 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 x;
    // this gets unrolled by 8
    for (size_t i = 0; i < n/4; i++) {
        x = _mm_add_ps(sum, m128_scan(v[i]));
        v[i] = x;
        sum = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
    return sum;
}

void NOINLINE scan_inplace(float* xs, size_t n) {
    scan_inplace_(xs, n, _mm_set1_ps(0));
}

__m128 scan_inplace_ss2_(float* xs, size_t n, __m128 sum) {
    assert(n % 8 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
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
    return sum;
}

void NOINLINE scan_inplace_ss2(float* xs, size_t n) {
    scan_inplace_ss2_(xs, n, _mm_set1_ps(0));
}

__m128 scan_inplace_ss4_(float* xs, size_t n, __m128 sum) {
    assert(n % 16 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
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
    return sum;
}

void NOINLINE scan_inplace_ss4(float* xs, size_t n) {
    scan_inplace_ss4_(xs, n, _mm_set1_ps(0));
}

void NOINLINE scan_unaligned_simple(float* xs, size_t n) {
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
    scan_inplace_ss4_(xs, n_aligned, _mm_set1_ps(valid ? xs[-1] : 0));
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

// handles all alignments and all n
void NOINLINE scan_unaligned(float* xs, size_t n) {
    if (n < 4) {
        scan_simple_(xs, n, 0);
        return;
    }

    int valid = 0;

    while (((uintptr_t)xs) & 0b1111) {
        if (valid) {
            *xs += *(xs-1);
        }
        xs += 1;
        n -= 1;
        valid = 1;
    }

    { // 16
        size_t n_aligned = (n / 16) * 16;
        /*printf(" n_aligned16=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_ss4_(xs, n_aligned, _mm_set1_ps(valid ? xs[-1] : 0));
        xs += n_aligned;
        n -= n_aligned;
        valid = valid || (n_aligned > 0);
    }
    { // 8
        size_t n_aligned = (n / 8) * 8;
        /*printf(" n_aligned8=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_ss2_(xs, n_aligned, _mm_set1_ps(valid ? xs[-1] : 0));
        xs += n_aligned;
        n -= n_aligned;
        valid = valid || (n_aligned > 0);
    }
    { // 4
        size_t n_aligned = (n / 4) * 4;
        /*printf(" n_aligned4=%ld valid=%d sum_in=%f\n", n_aligned, valid, valid ? xs[-1] : 0);*/
        scan_inplace_(xs, n_aligned, _mm_set1_ps(valid ? xs[-1] : 0));
        xs += n_aligned;
        n -= n_aligned;
    }
    // this is wrong for n < 3
    assert(n <= 3);
    /*printf(" n_rem=%ld\n", n);*/
    for (ssize_t i = 0; i < n; i++) {
        xs[i] += xs[i-1];
    }
}

void NOINLINE scan_unaligned2(float* xs, size_t n) {
    if (n < 4) {
        scan_simple_(xs, n, 0);
        return;
    }

    float sum = 0;

    size_t alignment = (((uintptr_t)xs) & 0b1111) / 4;
    /*printf("%p n=%ld alignment=%ld\n", xs, n, alignment);*/
    switch (alignment) {
        case 0: break;
        case 1: // xs[0] is alignment 4, xs[1] is 8, xs[2] is 12, xs[3] is 16
                xs[1] += xs[0];
                xs[2] += xs[1];
                sum = xs[2];
                xs += 3;
                n -= 3;
                break;
        case 2: // xs[0] is alignment 8, xs[1] is 12, xs[2] is 16
                xs[1] += xs[0];
                sum = xs[1];
                xs += 2;
                n -= 2;
                break;
        case 3: // xs[0] is alignment 12, xs[1] is 16
                sum = xs[0];
                xs += 1;
                n -= 1;
                break;
        default:
                __builtin_unreachable();
    }

    __m128 sumv = _mm_set1_ps(sum);

    { // 16
        size_t n_aligned = (n / 16) * 16;
        /*printf(" n_aligned16=%ld\n", n_aligned);*/
        sumv = scan_inplace_ss4_(xs, n_aligned, sumv);
        xs += n_aligned;
        n -= n_aligned;
    }
    { // 4
        __builtin_assume(n < 16);
        size_t n_aligned = (n / 4) * 4;
        /*printf(" n_aligned4=%ld\n", n_aligned);*/
        sumv = scan_inplace_(xs, n_aligned, sumv);
        xs += n_aligned;
        n -= n_aligned;
    }
    sum = _mm_cvtss_f32(sumv);
    switch (n) {
        case 0: break;
        case 1:
                xs[0] += sum;
                break;
        case 2:
                xs[0] += sum;
                xs[1] += xs[0];
                break;
        case 3:
                xs[0] += sum;
                xs[1] += xs[0];
                xs[2] += xs[1];
                break;
        default:
                __builtin_unreachable();
    }
}

void NOINLINE scan_unaligned3(float* xs, size_t n) {
    if (n < 16) {
        scan_simple_(xs, n, 0);
        return;
    }
    float sum = 0;

    size_t alignment = (((uintptr_t)xs) & 0b1111) / 4;
    /*printf("%p n=%ld alignment=%ld\n", xs, n, alignment);*/
    switch (alignment) {
        case 0: break;
        case 1: // xs[0] is alignment 4, xs[1] is 8, xs[2] is 12, xs[3] is 16
                xs[1] += xs[0];
                xs[2] += xs[1];
                sum = xs[2];
                xs += 3;
                n -= 3;
                break;
        case 2: // xs[0] is alignment 8, xs[1] is 12, xs[2] is 16
                xs[1] += xs[0];
                sum = xs[1];
                xs += 2;
                n -= 2;
                break;
        case 3: // xs[0] is alignment 12, xs[1] is 16
                sum = xs[0];
                xs += 1;
                n -= 1;
                break;
    }

    __m128 sumv = _mm_set1_ps(sum);

    { // 16
        size_t n_aligned = (n / 16) * 16;
        /*printf(" n_aligned16=%ld\n", n_aligned);*/
        sumv = scan_inplace_ss4_(xs, n_aligned, sumv);
        xs += n_aligned;
        n -= n_aligned;
    }
    // this can only go for 1 loop
    /*{ // 8*/
    /*    size_t n_aligned = (n / 8) * 8;*/
    /*    printf(" n_aligned8=%ld\n", n_aligned);*/
    /*    sumv = scan_inplace_ss2_(xs, n_aligned, sumv);*/
    /*    xs += n_aligned;*/
    /*    n -= n_aligned;*/
    /*}*/
    /*{ // 4*/
    /*    size_t n_aligned = (n / 4) * 4;*/
    /*    printf(" n_aligned4=%ld\n", n_aligned);*/
    /*    sumv = scan_inplace_(xs, n_aligned, sumv);*/
    /*    xs += n_aligned;*/
    /*    n -= n_aligned;*/
    /*}*/

    /*assert(n <= 3);*/
    /*printf(" n_rem=%ld\n", n);*/
    /*for (ssize_t i = 0; i < n; i++) {*/
    /*    xs[i] += xs[i-1];*/
    /*}*/
    // so confusing, dont use _mm_extract_ps
    scan_simple_(xs, n, _mm_cvtss_f32(sumv));

}

// handles %4 alignment and n%4 == 0
void NOINLINE scan_aligned_but_varying_size(float* xs, size_t n) {
    __m128 sumv = _mm_set1_ps(0);

    { // 16
        size_t n_aligned = (n / 16) * 16;
        /*printf(" n_aligned16=%ld\n", n_aligned);*/
        sumv = scan_inplace_ss4_(xs, n_aligned, sumv);
        xs += n_aligned;
        n -= n_aligned;
    }
    /*{ // 8*/
    /*    size_t n_aligned = (n / 8) * 8;*/
        /*printf(" n_aligned8=%ld\n", n_aligned);*/
    /*    sumv = scan_inplace_ss2_(xs, n_aligned, sumv);*/
    /*    xs += n_aligned;*/
    /*    n -= n_aligned;*/
    /*}*/
    { // 4
        size_t n_aligned = (n / 4) * 4;
        /*printf(" n_aligned4=%ld\n", n_aligned);*/
        sumv = scan_inplace_(xs, n_aligned, sumv);
        xs += n_aligned;
        n -= n_aligned;
    }
}

// this looks prettier, but the ending loop doesn't get unrolled
void NOINLINE scan_aligned_but_varying_size2(float* xs, size_t n) {
    __m128 sumv = _mm_set1_ps(0);

    while (n >= 16) {
        sumv = scan_inplace_ss4_(xs, 16, sumv);
        xs += 16;
        n -= 16;
    }

    while (n > 0) {
        sumv = scan_inplace_(xs, 4, sumv);
        xs += 4;
        n -= 4;
    }
}

void NOINLINE scan_aligned_but_varying_size3(float* xs, size_t n) {
    __m128 sumv = _mm_set1_ps(0);

    while (n >= 16) {
        sumv = scan_inplace_ss4_(xs, 16, sumv);
        xs += 16;
        n -= 16;
    }

    size_t n_aligned = (n / 4) * 4;
    sumv = scan_inplace_(xs, n_aligned, sumv);
}


void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", xs[i]);
    }
    printf("\n");
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
static void init_norm(float* xs, size_t N, float v) {
    float sum = 0;
    for (size_t i = 0; i < N; i++) { xs[i] = v; }
    for (size_t i = 0; i < N; i++) { sum += xs[i]; }
    for (size_t i = 0; i < N; i++) { xs[i] /= sum; }
}
#pragma clang diagnostic pop

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
        /*TEST(scan_aligned_but_varying_size);*/
        /*TEST(scan_aligned_but_varying_size2);*/
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
        // though this isn't perfect; since we aren't reallocating we aren't getting address sanitizer
        // checks that things aren't out of bounds
#define TESTU(name) \
        for (size_t N = 8; N <= 48; N += 8) { \
            for (size_t i = 0; i < 4; i++) { \
                for (size_t j = 0; j < 4; j++) { \
                    size_t n = N - i - j; \
                    init_norm(xs + i, n, 0.1); \
                    init_norm(ref + i, n, 0.1); \
                    scan_simple(ref + i, n); \
                    dump_array(xs + i, n); \
                    name(xs + i, n); \
                    dump_array(ref + i, n); \
                    dump_array(xs + i, n); \
                    diff = total_diff(ref + i, xs + i, n); \
                    printf("unaligned %s n=%ld i=%ld j=%ld total diff %f avg diff %e\n", STRINGIFY(name), n, i, j, diff, (double)diff / n); \
                    assert((double)diff / n <= MAX_AVG_DIFF); \
                } \
            } \
        } \
        for (size_t N = 1; N <= 8; N++) { \
            for (size_t i = 0; i < 4; i++) { \
                name(xs + i, N); \
            }\
        }

        TESTU(scan_unaligned_simple);
        TESTU(scan_unaligned);
        TESTU(scan_unaligned2);
        TESTU(scan_unaligned3);

#undef TESTU
        free(xs);
        free(ref);
    }
#else

    size_t rounds = 200000000;

    Timespec start, stop;

    /*for (size_t N = 8; N <= 512; N += 8) {*/
    for (size_t N = 8; N <= 128; N += 1) {
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
        if (N % 4 == 0) {
            BENCH(scan_inplace);
        }
        if (N % 8 == 0) {
            BENCH(scan_inplace_ss2);
        }
        if (N % 16 == 0) {
            BENCH(scan_inplace_ss4);
        }
        BENCH(scan_unaligned_simple);
        BENCH(scan_unaligned);
        BENCH(scan_unaligned2);
        BENCH(scan_unaligned3);
        // TODO I think these are buggy
        /*BENCH(scan_aligned_but_varying_size);*/
        /*BENCH(scan_aligned_but_varying_size2);*/
        /*BENCH(scan_aligned_but_varying_size3);*/

        free(xs);

#undef BENCH

    }
#endif
}
