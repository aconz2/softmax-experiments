// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "random.h"

#define STRINGIFY(x) #x
#define BILLION  1000000000LL
#define INLINE __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))

typedef struct timespec Timespec;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

static u64 elapsed_ns(Timespec start, Timespec stop) {
  return (u64)(stop.tv_sec - start.tv_sec) * BILLION + (u64)(stop.tv_nsec - start.tv_nsec);
}

#pragma clang diagnostic pop

int popcount32(u32 x) { return __builtin_popcount(x); }
int popcount64(u64 x) { return __builtin_popcountll(x); }

// NOTE: in general, these return in the range [0, n] inclusive
// when searching a cdf, we expect v to be [0, 1) and xs[n-1] to be 1.0
// but with numerical error, xs[n-1] might happen to be 0.998 or something and v to
// be 0.999 in which case we will get n out, which is nonsensical for sampling
// so consumers need to clamp the max value at n-1 to be safe
// two thoughts
//   1) set xs[n-1] to be INF or some "large" number like 10.0 so you always get [0, n)
//   2) just pass in n-1 as n so that you get [0, n-1] out instead
//      this just makes it slightly trickier since like ymm search will get confused seeing that
//      n is not a multiple of 8

// this is based on the musl
size_t binary_search_(float* xs, size_t n, float v) {
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

size_t NOINLINE binary_search(float* xs, size_t n, float v) {
    return binary_search_(xs, n, v);
}

// based on https://en.algorithmica.org/hpc/data-structures/binary-search/
size_t binary_search_alt(float* xs, size_t n, float v) {
    size_t l = 0, r = n - 1;
    while (l < r) {
        size_t m = (l + r) / 2;
        if (xs[m] >= v)
            r = m;
        else
            l = m + 1;
    }
    return l;
}

// based on https://en.algorithmica.org/hpc/data-structures/binary-search/
// clang19 not generating a cmov here
size_t binary_search_alt_branchless(float* xs, size_t n, float v) {
    float* cur = xs;
    size_t len = n;
    while (len > 1) {
        size_t half = len / 2;
        cur += (cur[half - 1] < v) * half; // will be replaced with a "cmov"
        len -= half;
    }
    return cur - xs;
}

size_t binary_search_alt_branchless_pftch(float* xs, size_t n, float v) {
    float* cur = xs;
    size_t len = n;
    while (len > 1) {
        size_t half = len / 2;
        len -= half;
        __builtin_prefetch(&cur[len / 2 - 1]);
        __builtin_prefetch(&cur[half + len / 2 - 1]);
        cur += (cur[half - 1] < v) * half; // will be replaced with a "cmov"
    }
    return cur - xs;
}


void NOINLINE binary_search2_easy(float* xs, size_t n, float v[2], size_t ret[2]) {
    for (size_t i = 0; i < 2; i++) {
        ret[i] = binary_search(xs, n, v[i]);
    }
}

void NOINLINE binary_search4_easy(float* xs, size_t n, float v[4], size_t ret[4]) {
    for (size_t i = 0; i < 4; i++) {
        ret[i] = binary_search(xs, n, v[i]);
    }
}

void NOINLINE binary_search8_easy(float* xs, size_t n, float v[8], size_t ret[8]) {
    for (size_t i = 0; i < 8; i++) {
        ret[i] = binary_search(xs, n, v[i]);
    }
}

// trying to do superscalar but this is looking uglier now
void NOINLINE binary_search2(float* xs, size_t n, float vs[2], size_t ret[2]) {
    float u = vs[0];
    float v = vs[1];
    float* ucur = xs;
    float* vcur = xs;
    size_t un = n, vn = n;
#define INNER(X) \
    if (X <= X##cur[X##n/2]) { \
        X##n /= 2; \
    } else { \
        X##cur = &X##cur[X##n/2 + 1]; \
        X##n -= X##n/2 + 1; \
    }

    while (un > 0 && vn > 0) {
        INNER(u)
        INNER(v)
    }
    while (un > 0) { INNER(u) }
    while (vn > 0) { INNER(v) }
    assert(un == 0);
    assert(vn == 0);
    ret[0] = ucur - xs;
    ret[1] = vcur - xs;
#undef INNER
}

void NOINLINE binary_search4(float* xs, size_t n, float vs[4], size_t ret[4]) {
    float u = vs[0];
    float v = vs[1];
    float w = vs[2];
    float z = vs[3];
    float* ucur = xs;
    float* vcur = xs;
    float* wcur = xs;
    float* zcur = xs;
    size_t un = n, vn = n, wn = n, zn = n;
#define INNER(X) \
    if (X <= X##cur[X##n/2]) { \
        X##n /= 2; \
    } else { \
        X##cur = &X##cur[X##n/2 + 1]; \
        X##n -= X##n/2 + 1; \
    }

    while (un > 0 && vn > 0 && wn > 0 && zn > 0) {
        INNER(u)
        INNER(v)
        INNER(w)
        INNER(z)
    }
    while (un > 0) { INNER(u) }
    while (vn > 0) { INNER(v) }
    while (wn > 0) { INNER(w) }
    while (zn > 0) { INNER(z) }
    ret[0] = ucur - xs;
    ret[1] = vcur - xs;
    ret[2] = wcur - xs;
    ret[3] = zcur - xs;
#undef INNER
}

void INLINE binary_search8_(float* xs, size_t N, float vs[8], size_t ret[8]) {
    float* cur[8] = {xs, xs, xs, xs, xs, xs, xs, xs};
    size_t n[8] = {N, N, N, N, N, N, N, N};
#define INNER(i) \
    if (vs[i] <= cur[i][n[i]/2]) { \
        n[i] /= 2; \
    } else { \
        cur[i] = &cur[i][n[i]/2 + 1]; \
        n[i] -= n[i]/2 + 1; \
    }

    while (n[0] > 0 && n[1] > 0 && n[2] > 0 && n[3] > 0 &&
            n[4] > 0 && n[5] > 0 && n[6] > 0 && n[7] > 0) {
        INNER(0)
        INNER(1)
        INNER(2)
        INNER(3)
        INNER(4)
        INNER(5)
        INNER(6)
        INNER(7)
    }
    while (n[0] > 0) { INNER(0) }
    while (n[1] > 0) { INNER(1) }
    while (n[2] > 0) { INNER(2) }
    while (n[3] > 0) { INNER(3) }
    while (n[4] > 0) { INNER(4) }
    while (n[5] > 0) { INNER(5) }
    while (n[6] > 0) { INNER(6) }
    while (n[7] > 0) { INNER(7) }
    for (int i = 0; i < 8; i++) {
        ret[i] = cur[i] - xs;
    }
#undef INNER
}

void NOINLINE binary_search8(float* xs, size_t N, float vs[8], size_t ret[8]) {
    binary_search8_(xs, N, vs, ret);
}

void NOINLINE binary_search8_256(float* xs, size_t N, float vs[8], size_t ret[8]) {
    (void)N;
    binary_search8_(xs, 256, vs, ret);
}

void binary_search9(float* xs, size_t N, float vs[9], size_t ret[9]) {
    float* cur[9] = {xs, xs, xs, xs, xs, xs, xs, xs, xs};
    size_t n[9] = {N, N, N, N, N, N, N, N, N};
#define INNER(i) \
    if (vs[i] <= cur[i][n[i]/2]) { \
        n[i] /= 2; \
    } else { \
        cur[i] = &cur[i][n[i]/2 + 1]; \
        n[i] -= n[i]/2 + 1; \
    }

    while (n[0] > 0 && n[1] > 0 && n[2] > 0 && n[3] > 0 &&
            n[4] > 0 && n[5] > 0 && n[6] > 0 && n[7] > 0 &&
            n[8] > 0
            ) {
        INNER(0)
        INNER(1)
        INNER(2)
        INNER(3)
        INNER(4)
        INNER(5)
        INNER(6)
        INNER(7)
        INNER(8)
    }
    while (n[0] > 0) { INNER(0) }
    while (n[1] > 0) { INNER(1) }
    while (n[2] > 0) { INNER(2) }
    while (n[3] > 0) { INNER(3) }
    while (n[4] > 0) { INNER(4) }
    while (n[5] > 0) { INNER(5) }
    while (n[6] > 0) { INNER(6) }
    while (n[7] > 0) { INNER(7) }
    while (n[8] > 0) { INNER(8) }
    for (int i = 0; i < 9; i++) {
        ret[i] = cur[i] - xs;
    }
#undef INNER
}

void binary_search10(float* xs, size_t N, float vs[10], size_t ret[10]) {
    float* cur[10] = {xs, xs, xs, xs, xs, xs, xs, xs, xs, xs};
    size_t n[10] = {N, N, N, N, N, N, N, N, N, N};
#define INNER(i) \
    if (vs[i] <= cur[i][n[i]/2]) { \
        n[i] /= 2; \
    } else { \
        cur[i] = &cur[i][n[i]/2 + 1]; \
        n[i] -= n[i]/2 + 1; \
    }

    while (n[0] > 0 && n[1] > 0 && n[2] > 0 && n[3] > 0 &&
            n[4] > 0 && n[5] > 0 && n[6] > 0 && n[7] > 0 &&
            n[8] > 0 && n[9] > 0
            ) {
        INNER(0)
        INNER(1)
        INNER(2)
        INNER(3)
        INNER(4)
        INNER(5)
        INNER(6)
        INNER(7)
        INNER(8)
        INNER(9)
    }
    while (n[0] > 0) { INNER(0) }
    while (n[1] > 0) { INNER(1) }
    while (n[2] > 0) { INNER(2) }
    while (n[3] > 0) { INNER(3) }
    while (n[4] > 0) { INNER(4) }
    while (n[5] > 0) { INNER(5) }
    while (n[6] > 0) { INNER(6) }
    while (n[7] > 0) { INNER(7) }
    while (n[8] > 0) { INNER(8) }
    while (n[9] > 0) { INNER(9) }
    for (int i = 0; i < 10; i++) {
        ret[i] = cur[i] - xs;
    }
#undef INNER
}

void binary_search12(float* xs, size_t N, float vs[12], size_t ret[12]) {
    float* cur[12] = {xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs};
    size_t n[12] = {N, N, N, N, N, N, N, N, N, N, N, N};
#define INNER(i) \
    if (vs[i] <= cur[i][n[i]/2]) { \
        n[i] /= 2; \
    } else { \
        cur[i] = &cur[i][n[i]/2 + 1]; \
        n[i] -= n[i]/2 + 1; \
    }

    while (n[0] > 0 && n[1] > 0 && n[2] > 0 && n[3] > 0 &&
            n[4] > 0 && n[5] > 0 && n[6] > 0 && n[7] > 0 &&
            n[8] > 0 && n[9] > 0 && n[10] > 0 && n[11] > 0
            ) {
        INNER(0)
        INNER(1)
        INNER(2)
        INNER(3)
        INNER(4)
        INNER(5)
        INNER(6)
        INNER(7)
        INNER(8)
        INNER(9)
        INNER(10)
        INNER(11)
    }
    while (n[0] > 0) { INNER(0) }
    while (n[1] > 0) { INNER(1) }
    while (n[2] > 0) { INNER(2) }
    while (n[3] > 0) { INNER(3) }
    while (n[4] > 0) { INNER(4) }
    while (n[5] > 0) { INNER(5) }
    while (n[6] > 0) { INNER(6) }
    while (n[7] > 0) { INNER(7) }
    while (n[8] > 0) { INNER(8) }
    while (n[9] > 0) { INNER(9) }
    while (n[10] > 0) { INNER(10) }
    while (n[11] > 0) { INNER(11) }
    for (int i = 0; i < 12; i++) {
        ret[i] = cur[i] - xs;
    }
#undef INNER
}

void binary_search16(float* xs, size_t N, float vs[16], size_t ret[16]) {
    float* cur[16] = {xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs, xs};
    size_t n[16] = {N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N};
#define INNER(i) \
    if (vs[i] <= cur[i][n[i]/2]) { \
        n[i] /= 2; \
    } else { \
        cur[i] = &cur[i][n[i]/2 + 1]; \
        n[i] -= n[i]/2 + 1; \
    }

    while (n[0] > 0 && n[1] > 0 && n[2] > 0 && n[3] > 0 &&
            n[4] > 0 && n[5] > 0 && n[6] > 0 && n[7] > 0 &&
            n[8] > 0 && n[9] > 0 && n[10] > 0 && n[11] > 0 &&
            n[12] > 0 && n[13] > 0 && n[14] > 0 && n[15] > 0
            ) {
        INNER(0)
        INNER(1)
        INNER(2)
        INNER(3)
        INNER(4)
        INNER(5)
        INNER(6)
        INNER(7)
        INNER(8)
        INNER(9)
        INNER(10)
        INNER(11)
        INNER(12)
        INNER(13)
        INNER(14)
        INNER(15)
    }
    while (n[0] > 0) { INNER(0) }
    while (n[1] > 0) { INNER(1) }
    while (n[2] > 0) { INNER(2) }
    while (n[3] > 0) { INNER(3) }
    while (n[4] > 0) { INNER(4) }
    while (n[5] > 0) { INNER(5) }
    while (n[6] > 0) { INNER(6) }
    while (n[7] > 0) { INNER(7) }
    while (n[8] > 0) { INNER(8) }
    while (n[9] > 0) { INNER(9) }
    while (n[10] > 0) { INNER(10) }
    while (n[11] > 0) { INNER(11) }
    while (n[12] > 0) { INNER(12) }
    while (n[13] > 0) { INNER(13) }
    while (n[14] > 0) { INNER(14) }
    while (n[15] > 0) { INNER(15) }
    for (int i = 0; i < 16; i++) {
        ret[i] = cur[i] - xs;
    }
#undef INNER
}

size_t NOINLINE linear_search(float* xs, size_t n, float v) {
    for (size_t i = 0; i < n; i++) {
        if (v <= xs[i]) {
            return i;
        }
    }
    return n;
}

// assumes no NaN
size_t NOINLINE xmm_search(float* xs, size_t n, float needle) {
    if (n < 16) return binary_search(xs, n, needle);
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

size_t ymm_search_(float* xs, size_t n, float needle) {
    if (n < 16) return binary_search(xs, n, needle);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c;
    int m;
    for (size_t i = 0; i < n/8; i++) {
        // less than or equal, quiet
        c = _mm256_cmp_ps(needlev, v[i], _CMP_LE_OQ);
        m = _mm256_movemask_ps(c);
        if (m > 0) {
            // this gets the least significant set bit, so if more than one matches, we are okay
            int index = __builtin_ctz(m);
            return i*8 + index;
        }
    }
    return n;
}

size_t NOINLINE ymm_search(float* xs, size_t n, float needle) {
    return ymm_search_(xs, n, needle);
}

size_t NOINLINE ymm2_search(float* xs, size_t n, float needle) {
    if (n < 16) return binary_search(xs, n, needle);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c1, c2;
    int m1, m2;
    for (size_t i = 0; i < n/8; i += 2) {
        // less than or equal, quiet
        c1 = _mm256_cmp_ps(needlev, v[i], _CMP_LE_OQ);
        c2 = _mm256_cmp_ps(needlev, v[i+1], _CMP_LE_OQ);
        m1 = _mm256_movemask_ps(c1);
        m2 = _mm256_movemask_ps(c2);
        if (m1 > 0) {
            // this gets the least significant set bit, so if more than one matches, we are okay
            int index = __builtin_ctz(m1);
            return i*8 + index;
        }
        if (m2 > 0) {
            int index = __builtin_ctz(m2);
            return (i+1)*8 + index;
        }
    }
    return n;
}

size_t NOINLINE ymm_search_16(float* xs, size_t n, float needle) {
    __builtin_assume(n == 16);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c0, c1;
    int m1, m2;
    c0 = _mm256_cmp_ps(needlev, v[0], _CMP_LE_OQ);
    c1 = _mm256_cmp_ps(needlev, v[1], _CMP_LE_OQ);
    m1 = _mm256_movemask_ps(c0);
    m2 = _mm256_movemask_ps(c1);
    return 16 - (popcount32(m1) + popcount32(m2));
}

size_t NOINLINE ymm_search_16_gt(float* xs, size_t n, float needle) {
    __builtin_assume(n == 16);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c0, c1;
    int m0, m1;
    c0 = _mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ);
    c1 = _mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ);
    m0 = _mm256_movemask_ps(c0);
    m1 = _mm256_movemask_ps(c1);
    return popcount32(m0) + popcount32(m1);
}

size_t NOINLINE ymm_search_32_gt(float* xs, size_t n, float needle) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c0, c1, c2, c3;
    int m0, m1, m2, m3;
    c0 = _mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ);
    c1 = _mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ);
    c2 = _mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ);
    c3 = _mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ);
    m0 = _mm256_movemask_ps(c0);
    m1 = _mm256_movemask_ps(c1);
    m2 = _mm256_movemask_ps(c2);
    m3 = _mm256_movemask_ps(c3);
    return popcount32(m0) + popcount32(m1) + popcount32(m2) + popcount32(m3);
}

size_t NOINLINE ymm_search_32_gt_or(float* xs, size_t n, float needle) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c;
    u64 ret = 0;
    for (size_t i = 0; i < 32/8; i++) {
        c = _mm256_cmp_ps(needlev, v[i], _CMP_GT_OQ);
        ret |= _mm256_movemask_ps(c);
        if (i != 3) {
            ret <<= 8;
        }
    }
    return popcount32(ret);
}

size_t NOINLINE ymm_search_32_gt_or_tree(float* xs, size_t n, float needle) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    u64 a, b, c, d;
    a = _mm256_movemask_ps(_mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ));
    b = _mm256_movemask_ps(_mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ));
    a |= b << 8;
    c = _mm256_movemask_ps(_mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ));
    d = _mm256_movemask_ps(_mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ));
    c |= d << 8;
    a |= c << 16;
    return popcount32(a);
}

void NOINLINE ymm_search_32_gt_ss2(float* xs, size_t n, float needles[2], size_t ret[2]) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev0 = _mm256_set1_ps(needles[0]);
    __m256 needlev1 = _mm256_set1_ps(needles[1]);
    __m256 c0, c1;
    ret[0] = 0;
    ret[1] = 0;
    for (size_t i = 0; i < 32/8; i++) {
        c0 = _mm256_cmp_ps(needlev0, v[i], _CMP_GT_OQ);
        c1 = _mm256_cmp_ps(needlev1, v[i], _CMP_GT_OQ);
        ret[0] += popcount32(_mm256_movemask_ps(c0));
        ret[1] += popcount32(_mm256_movemask_ps(c1));
    }
}

size_t NOINLINE ymm_search_64_gt(float* xs, size_t n, float needle) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;
    int m0, m1, m2, m3, m4, m5, m6, m7;
    c0 = _mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ);
    c1 = _mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ);
    c2 = _mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ);
    c3 = _mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ);
    c4 = _mm256_cmp_ps(needlev, v[4], _CMP_GT_OQ);
    c5 = _mm256_cmp_ps(needlev, v[5], _CMP_GT_OQ);
    c6 = _mm256_cmp_ps(needlev, v[6], _CMP_GT_OQ);
    c7 = _mm256_cmp_ps(needlev, v[7], _CMP_GT_OQ);
    m0 = _mm256_movemask_ps(c0);
    m1 = _mm256_movemask_ps(c1);
    m2 = _mm256_movemask_ps(c2);
    m3 = _mm256_movemask_ps(c3);
    m4 = _mm256_movemask_ps(c4);
    m5 = _mm256_movemask_ps(c5);
    m6 = _mm256_movemask_ps(c6);
    m7 = _mm256_movemask_ps(c7);
    return popcount32(m0) + popcount32(m1) + popcount32(m2) + popcount32(m3) + popcount32(m4) + popcount32(m5) + popcount32(m6) + popcount32(m7);
}

size_t NOINLINE ymm_search_64_gt_interleave(float* xs, size_t n, float needle) {
    __builtin_assume(n == 32);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;
    size_t ret = 0;
    c0 = _mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c0));
    c1 = _mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c1));
    c2 = _mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c2));
    c3 = _mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c3));
    c4 = _mm256_cmp_ps(needlev, v[4], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c4));
    c5 = _mm256_cmp_ps(needlev, v[5], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c5));
    c6 = _mm256_cmp_ps(needlev, v[6], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c6));
    c7 = _mm256_cmp_ps(needlev, v[7], _CMP_GT_OQ);
    ret += popcount32(_mm256_movemask_ps(c7));
    return ret;
}

size_t NOINLINE ymm_search_64_gt_loop(float* xs, size_t n, float needle) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c[8];
    for (size_t i = 0; i < 8; i++) {
        c[i] = _mm256_cmp_ps(needlev, v[i], _CMP_GT_OQ);
    }
    size_t ret = 0;
    for (size_t i = 0; i < 8; i++) {
        ret += popcount32(_mm256_movemask_ps(c[i]));
    }
    return ret;
}

// this is better than the above and the same as manaul interleave but easier to write
size_t NOINLINE ymm_search_64_gt_loop_interleave(float* xs, size_t n, float needle) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c;
    size_t ret = 0;
    for (size_t i = 0; i < 64/8; i++) {
        c = _mm256_cmp_ps(needlev, v[i], _CMP_GT_OQ);
        ret += popcount32(_mm256_movemask_ps(c));
    }
    return ret;
}

// this one isn't working, the compiler is noticing something is zero and is only doing 3 things
// same format works for 32 and should be able to get 8 masks in one u64 so idk
size_t NOINLINE ymm_search_64_gt_or(float* xs, size_t n, float needle) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c;
    u64 ret = 0;
    for (size_t i = 0; i < 64/8; i++) {
        c = _mm256_cmp_ps(needlev, v[i], _CMP_GT_OQ);
        ret |= _mm256_movemask_ps(c);
        if (i != 7) {
            ret <<= 8;
        }
    }
    return popcount64(ret);
}

u64 movemask64(__m256 x) { return _mm256_movemask_ps(x); }

size_t ymm_search_64_gt_or_tree_(float* xs, size_t n, float needle) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    u64 a = 0, b = 0, c = 0, d = 0;

    a = movemask64(_mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ)) << 0;
    b = movemask64(_mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ)) << 8;
    c = movemask64(_mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ)) << 16;
    d = movemask64(_mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ)) << 24;

    a |= movemask64(_mm256_cmp_ps(needlev, v[4], _CMP_GT_OQ)) << 32;
    b |= movemask64(_mm256_cmp_ps(needlev, v[5], _CMP_GT_OQ)) << 40;
    c |= movemask64(_mm256_cmp_ps(needlev, v[6], _CMP_GT_OQ)) << 48;
    d |= movemask64(_mm256_cmp_ps(needlev, v[7], _CMP_GT_OQ)) << 56;

    a |= b;
    c |= d;

    a |= c;
    return popcount64(a);
}

size_t NOINLINE ymm_search_64_gt_or_tree(float* xs, size_t n, float needle) {
    return ymm_search_64_gt_or_tree_(xs, n, needle);
}

size_t NOINLINE ymm_search_64_gt_or_tree2(float* xs, size_t n, float needle) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    u64 a = 0, b = 0, c = 0, d = 0;

    a = movemask64(_mm256_cmp_ps(needlev, v[0], _CMP_GT_OQ)) << 8;
    b = movemask64(_mm256_cmp_ps(needlev, v[1], _CMP_GT_OQ)) << 8;
    c = movemask64(_mm256_cmp_ps(needlev, v[2], _CMP_GT_OQ)) << 8;
    d = movemask64(_mm256_cmp_ps(needlev, v[3], _CMP_GT_OQ)) << 8;

    a |= movemask64(_mm256_cmp_ps(needlev, v[4], _CMP_GT_OQ));
    b |= movemask64(_mm256_cmp_ps(needlev, v[5], _CMP_GT_OQ));
    c |= movemask64(_mm256_cmp_ps(needlev, v[6], _CMP_GT_OQ));
    d |= movemask64(_mm256_cmp_ps(needlev, v[7], _CMP_GT_OQ));

    a |= b << 16;
    c |= d << 16;

    a |= c << 32;
    return popcount64(a);
}

void NOINLINE ymm_search_64_gt_ss2(float* xs, size_t n, float needles[2], size_t ret[2]) {
    __builtin_assume(n == 64);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev0 = _mm256_set1_ps(needles[0]);
    __m256 needlev1 = _mm256_set1_ps(needles[1]);
    __m256 c0, c1;
    ret[0] = 0;
    ret[1] = 0;
    for (size_t i = 0; i < 64/8; i++) {
        c0 = _mm256_cmp_ps(needlev0, v[i], _CMP_GT_OQ);
        c1 = _mm256_cmp_ps(needlev1, v[i], _CMP_GT_OQ);
        ret[0] += popcount32(_mm256_movemask_ps(c0));
        ret[1] += popcount32(_mm256_movemask_ps(c1));
    }
}

size_t NOINLINE ymm_search_128_gt(float* xs, size_t n, float needle) {
    __builtin_assume(n == 128);
    __m256* v = (__m256*)__builtin_assume_aligned(xs, 32);
    __m256 needlev = _mm256_set1_ps(needle);
    __m256 c;
    size_t ret = 0;
    for (size_t i = 0; i < 128/8; i++) {
        c = _mm256_cmp_ps(needlev, v[i], _CMP_GT_OQ);
        ret += popcount32(_mm256_movemask_ps(c));
    }
    return ret;
}

size_t ymm_search_128_gt_or_tree_(float* xs, size_t n, float needle) {
    return ymm_search_64_gt_or_tree_(xs, n, needle) + ymm_search_64_gt_or_tree_(xs + 64, n, needle);
}

size_t NOINLINE ymm_search_128_gt_or_tree(float* xs, size_t n, float needle) {
    return ymm_search_128_gt_or_tree_(xs, n, needle);
}

size_t ymm_search_256(float* xs, size_t N, float needle) {
    (void)N;
    return ymm_search_(xs, 256, needle);
}

size_t ymm_search_128(float* xs, size_t N, float needle) {
    (void)N;
    return ymm_search_(xs, 128, needle);
}

size_t NOINLINE ymm_search_256_gt(float* xs, size_t N, float needle) {
    return ymm_search_128_gt_or_tree_(xs, N, needle) + ymm_search_128_gt_or_tree_(xs+128, N, needle);
}

size_t NOINLINE ymm_search_256_binary1(float* xs, size_t N, float needle) {
    (void)N;
    if (needle <= xs[128]) {
        return ymm_search_128_gt_or_tree_(xs, N, needle);
    } else {
        return 128 + ymm_search_128_gt_or_tree_(xs + 128, N, needle);
    }
}

size_t NOINLINE ymm_search_256_binary2(float* xs, size_t N, float needle) {
    (void)N;
    if (needle <= xs[128]) {
        if (needle <= xs[64]) {
            return ymm_search_64_gt_or_tree_(xs, N, needle);
        } else {
            return 64 + ymm_search_64_gt_or_tree_(xs + 64, N, needle);
        }
    } else {
        if (needle <= xs[192]) {
            return 128 + ymm_search_64_gt_or_tree_(xs + 128, N, needle);
        } else {
            return 192 + ymm_search_64_gt_or_tree_(xs + 192, N, needle);
        }
    }
}

// these are buggy and a bit meh in initial perf so not trying further

/*size_t binary_search_ymm_16(float* xs, size_t n, float v) {*/
/*    float* cur = xs;*/
/*    while (n > 32) {*/
/*        if (v <= cur[n/2]) {*/
/*            n /= 2;*/
/*        } else {*/
/*            cur = &cur[n/2 + 1];*/
/*            n -= n/2 + 1;*/
/*        }*/
/*    }*/
/*    size_t offset = ymm_search(cur, 16, v);*/
/*    return cur - xs + offset;*/
/*}*/

/*size_t binary_search_xmm_16(float* xs, size_t n, float v) {*/
/*    float* cur = xs;*/
/*    while (n > 16) {*/
/*        if (v <= cur[n/2]) {*/
/*            n /= 2;*/
/*        } else {*/
/*            cur = &cur[n/2 + 1];*/
/*            n -= n/2 + 1;*/
/*        }*/
/*    }*/
/*    size_t offset = ((uintptr_t)cur & 0b1111) == 0 ? xmm_search(cur, 16, v) : linear_search(cur, 16, v);*/
/**/
/*    return cur - xs + offset;*/
/*}*/

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
#ifdef RUNTEST

    {
        size_t ret;
        float v;
        const float offset = 0.0001;
        for (size_t N = 8; N <= 256; N++) {
            float* xs = aligned_alloc(32, round_up_size_t(sizeof(float)*N, 32));
            init_cdf(xs, N, 0.1);
            /*dump_array(xs, N);*/

            for (size_t i = 0; i < N; i++) {
#define TEST(name) \
                v = xs[i]; ret = name(xs, N, v); \
                if (i != ret) { printf("%s fail for offset=0 N=%ld i=%ld v=%f exepected %ld got %ld\n", STRINGIFY(name), N, i, v, i, ret); } \
                assert(i == ret); \
                v = xs[i] - offset; ret = name(xs, N, v); \
                if (i != ret) { printf("%s fail for offset=- N=%ld i=%ld v=%f exepected %ld got %ld\n", STRINGIFY(name), N, i, v, i, ret); } \
                assert(i == ret); \
                v = xs[i] + offset; ret = name(xs, N, v); \
                if (i+1 != ret) { dump_array(xs, N); printf("%s fail offset=+ for N=%ld i=%ld v=%f exepected %ld got %ld\n", STRINGIFY(name), N, i, v, i+1, ret); } \
                assert(i+1 == ret);

                TEST(binary_search);
                TEST(linear_search);

                if (__builtin_popcount(N) == 1) {
                    TEST(xmm_search);
                    TEST(ymm_search);
                    TEST(ymm2_search);
                    if (N >= 16) {
                        /*TEST(binary_search_ymm_16);*/
                        /*TEST(binary_search_xmm_16);*/
                    }
                    if (N == 16) {
                        TEST(ymm_search_16);
                        TEST(ymm_search_16_gt);
                    }
                    if (N == 32) {
                        TEST(ymm_search_32_gt);
                        TEST(ymm_search_32_gt_or);
                    }
                    if (N == 64) {
                        TEST(ymm_search_64_gt);
                        /*TEST(ymm_search_64_gt_or);*/
                        TEST(ymm_search_64_gt_or_tree);
                        TEST(ymm_search_64_gt_or_tree2);
                        TEST(ymm_search_64_gt_loop);
                        TEST(ymm_search_64_gt_interleave);
                        TEST(ymm_search_64_gt_loop_interleave);
                    }
                    if (N == 128) {
                        TEST(ymm_search_128_gt);
                        TEST(ymm_search_128_gt_or_tree);
                    }
                    if (N == 256) {
                        TEST(ymm_search_256_gt);
                        TEST(ymm_search_256_binary1);
                        TEST(ymm_search_256_binary2);
                    }
                }


                {
                    size_t ret[2];
                    float v[2];
                    v[0] = xs[i];
                    v[1] = xs[i] - offset;
                    binary_search2(xs, N, v, ret);
                    assert(i == ret[0]);
                    assert(i == ret[1]);
                    v[0] = xs[i];
                    v[1] = xs[N-i-1];
                    binary_search2(xs, N, v, ret);
                    /*printf("i=%ld N-i-1=%ld xs[i]=%.2f xs[N-i-1]=%.2f ret is %ld %ld\n", i, N-i, xs[i], xs[N-i-1], ret[0], ret[1]);*/
                    assert(i == ret[0]);
                    assert(N-i-1 == ret[1]);
                    v[0] = xs[i] + offset;
                    v[1] = xs[N-i-1] + offset;
                    /*printf("searching %.2f %.2f\n", v[0], v[1]);*/
                    binary_search2(xs, N, v, ret);
                    /*printf("got %ld %ld expected %ld %ld\n", ret[0], ret[1], i+1, N-1);*/
                    /*printf("i=%ld %.2f ret is %ld %ld\n", i, xs[i] + 0.01, ret[0], ret[1]);*/
                    assert(i+1 == ret[0]);
                    assert(N-i-1 + 1 == ret[1]);
                }
            }

            free(xs);
        }
    }

#else

    Timespec start, stop;

    {
        size_t rounds = 10000000;
        PRNG32RomuQuad rng;
        rng_init(&rng, 42);
        u32 check = 0;
        clock_ns(&start);
        for (size_t i = 0; i < rounds; i++) {
            float x = rng_float(&rng);
            u32 y;
            memcpy(&y, &x, sizeof(float));
            check += y;
        }
        clock_ns(&stop);
        printf("RNG  %.2f ns/call %.2f ms check=%x\n", (double)elapsed_ns(start, stop) / (double)rounds, (double)elapsed_ns(start, stop) / 1000000, check);
    }

    size_t rounds = 3600000;
    /*if (rounds % 12 != 0) { // must be true so BENCHK is correct*/
    /*    return 1;*/
    /*}*/
    size_t check;

    for (size_t N = 8; N <= 512; N *= 2) {
        float* xs = aligned_alloc(32, sizeof(float)*N);
        init_cdf(xs, N, 0.1);

        PRNG32RomuQuad rng;

        printf("N=%ld\n", N);

#define BENCH(name) \
        rng_init(&rng, 42); \
        check = 0; \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds; i++) { \
            check += name(xs, N, rng_float(&rng)); \
        } \
        clock_ns(&stop); \
        printf("  %30s %.2f ns/call %.2f ms check=%lx\n", STRINGIFY(name), (double)elapsed_ns(start, stop) / (double)rounds, (double)elapsed_ns(start, stop) / 1000000, check);

#define BENCHK(K, name) \
        if (rounds % K != 0) { printf("rounds must be mod %d\n", K); return 1; } \
        rng_init(&rng, 42); \
        check = 0; \
        clock_ns(&start); \
        for (size_t i = 0; i < rounds/K; i++) { \
            size_t ret[K]; \
            float v[K]; \
            for (size_t k = 0; k < K; k++) { v[k] = rng_float(&rng); } \
            name(xs, N, v, ret); \
            for (size_t k = 0; k < K; k++) { check += ret[k]; } \
        } \
        clock_ns(&stop); \
        printf("  %30s %.2f ns/call %.2f ms check=%lx\n", STRINGIFY(name), (double)elapsed_ns(start, stop) / (double)rounds, (double)elapsed_ns(start, stop) / 1000000, check);

        if (N == 8) {
            BENCH(linear_search);
            BENCH(binary_search);
            BENCH(xmm_search);
            BENCH(ymm_search);
            continue;
        }
        BENCH(linear_search);
        BENCH(binary_search);
        // these didn't generate the cmov the article expected, and binary_search already does
        /*BENCH(binary_search_alt);*/
        /*BENCH(binary_search_alt_branchless);*/
        /*BENCH(binary_search_alt_branchless_pftch);*/
        BENCH(xmm_search);
        BENCH(ymm_search);
        BENCH(ymm2_search);
        /*BENCH(binary_search_ymm_16);*/
        /*BENCH(binary_search_xmm_16);*/

        BENCHK(2, binary_search2);
        BENCHK(2, binary_search2_easy);

        BENCHK(4, binary_search4_easy);
        BENCHK(4, binary_search4);

        BENCHK(8, binary_search8_easy);
        BENCHK(8, binary_search8);

        BENCHK(9, binary_search9);
        BENCHK(10, binary_search10);
        BENCHK(12, binary_search12);
        BENCHK(16, binary_search16);

        if (N == 16) {
            BENCH(ymm_search_16);
            BENCH(ymm_search_16_gt);
        }
        if (N == 32) {
            BENCH(ymm_search_32_gt);
            BENCH(ymm_search_32_gt_or);
            BENCH(ymm_search_32_gt_or_tree);
            BENCHK(2, ymm_search_32_gt_ss2);
        }
        if (N == 64) {
            BENCH(ymm_search_64_gt);
            BENCH(ymm_search_64_gt_or);
            BENCH(ymm_search_64_gt_or_tree);
            BENCH(ymm_search_64_gt_or_tree2);
            BENCH(ymm_search_64_gt_loop);
            BENCH(ymm_search_64_gt_loop_interleave);
            BENCH(ymm_search_64_gt_interleave);
            BENCHK(2, ymm_search_64_gt_ss2);
        }

        if (N == 128) {
            BENCH(ymm_search_128_gt);
            BENCH(ymm_search_128_gt_or_tree);
            BENCH(ymm_search_128);
        }
        if (N == 256) {
            BENCH(ymm_search_256);
            BENCH(ymm_search_256_gt);
            BENCHK(8, binary_search8_256);
            BENCH(ymm_search_256_binary1);
            BENCH(ymm_search_256_binary2);
        }

#undef BENCH
#undef BENCHK

        free(xs);
    }

#endif

}
