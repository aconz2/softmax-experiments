#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <assert.h>
#include <time.h>

typedef struct timespec Timespec;

#define BILLION  1000000000LL

#define STRINGIFY(x) #x

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

typedef uint64_t u64;

static u64 elapsed_ns(Timespec start, Timespec stop) {
  u64 acpre = (u64)(stop.tv_sec - start.tv_sec) * BILLION
               + (u64)(stop.tv_nsec - start.tv_nsec);
  return acpre;
}

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

void scan_inplace(float* xs, size_t n) {
    assert(n % 4 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_epi32(0);
    __m128 x;
    // this gets unrolled by 8
    for (size_t i = 0; i < n/4; i++) {
        x = _mm_add_ps(sum, m128_scan(v[i]));
        v[i] = x;
        sum = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

void scan_inplace_ss2(float* xs, size_t n) {
    assert(n % 8 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_epi32(0);
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

void scan_inplace_ss4(float* xs, size_t n) {
    assert(n % 16 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_epi32(0);
    __m128 a, b, c, d, sa, sb, sc;
    for (size_t i = 0; i < n/4; i += 4) {
        a = _mm_add_ps(sum, m128_scan(v[i]));

        b = m128_scan(v[i + 1]); // hgfe gfe fe e
        c = m128_scan(v[i + 2]); // lkji
        d = m128_scan(v[i + 3]); // ponm

        // 4 adds, 3 shuffles
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

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", xs[i]);
    }
    printf("\n");
}

int main() {
    size_t rounds = 1000000000;

    Timespec start, stop;
    if (0) {
        size_t N = 32;
        float* xs = aligned_alloc(16, sizeof(float)*N);
        for (size_t i = 0; i < N; i++) { xs[i] = i * 0.1; }
        dump_array(xs, N);

#define TEST(name) \
        name(xs, N); \
        dump_array(xs, N); \
        for (size_t i = 0; i < N; i++) { xs[i] = i * 0.1; }

        TEST(scan_simple);
        TEST(scan_inplace);
        TEST(scan_inplace_ss2);
        TEST(scan_inplace_ss4);

#undef TEST

        free(xs);
    }

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

        free(xs);

#undef BENCH

    }
}
