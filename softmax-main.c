// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include "sleefreduxavx2.h"

static __m128 m128_scan(__m128 x) {
    // d    c   b  a
    // c    b   a  0 +
    // dc   cb  ba a
    // ba   a   0  0 +
    // dcba cba ba a
    x = _mm_add_ps(x, (__m128)_mm_slli_si128((__m128i)x, 4));
    // compiler chooses a movelh with zero
    x = _mm_add_ps(x, (__m128)_mm_slli_si128((__m128i)x, 8));
    return x;
}

static __m128 m256_hadd_m128(__m256 x) {
    // h  g  f  e
    // d  b  c  a +
    // hd gb fc ea +
    // hdgbfcea gbfcea fcea ea
    // hdgbfcea{4}
    __m128 sum = _mm_add_ps(
        _mm256_extractf128_ps(x, 0),
        _mm256_extractf128_ps(x, 1)
        );
    sum = m128_scan(sum);
    sum = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3));
    return sum;
}

static __m256 m256_hadd(__m256 x) {
    __m128 sum = m256_hadd_m128(x);
    return _mm256_set_m128(sum, sum);
}

// scans within the two xmm lanes
static void m256_scan_partial(__m256 x, __m128* a, __m128* b) {
    // h g f e d c b a
    // hgfe  gfe  fe   e    |  dcba  cba   ba   a
    //           b          |            a
    x = _mm256_add_ps(x, (__m256)_mm256_slli_si256((__m256i)x, 4));
    x = _mm256_add_ps(x, (__m256)_mm256_slli_si256((__m256i)x, 8));
    *a = _mm256_extractf128_ps(x, 0);
    *b = _mm256_extractf128_ps(x, 1);
}

void softmax_temp(float* restrict src, float* restrict dst, size_t N, int dotemp, float temp) {
    /*__builtin_assume(N >= 8);*/
    __m256* restrict srcv = __builtin_assume_aligned(src, 32);
    __m256* restrict dstv = __builtin_assume_aligned(dst, 32);
    __m256 sum = _mm256_set1_ps(0);
    __m128 presum = _mm_set1_ps(0);
    __m256 x, x0, x1;
    __m128 a, b, c, d, sa, sb, sc;

    // tiny bump ~ .01ns with unrolling but the tail handling generates bad looking code
#pragma unroll 1
    for (size_t i = 0; i < N/8; i++) {
        // use Sleef_redux_finz_expf8_u10avx2 if your inputs might not be
        // in the approx range [-100, 100]
        x = Sleef_redux_expf8_u10avx2(dotemp ? _mm256_div_ps(srcv[i], _mm256_set1_ps(temp)) : srcv[i]);
        sum = _mm256_add_ps(sum, x);
        dstv[i] = x;
    }

    if (N == 8) {
        float tot = _mm_cvtss_f32(m256_hadd_m128(sum));
        dst[0] /= tot;
        // this becomes an unrolled fmadd
        for (size_t i = 1; i < N; i++) {
            dst[i] = dst[i - 1] + dst[i] / tot;
        }
        return;
    }

    sum = m256_hadd(sum);

    for (size_t i = 0; i < N/8; i += 2) {
        x0 = _mm256_div_ps(dstv[i+0], sum);
        x1 = _mm256_div_ps(dstv[i+1], sum);

        m256_scan_partial(x0, &a, &b);
        m256_scan_partial(x1, &c, &d);

        // a: dcba cba ba a
        // b: hgfe gfe fe e
        // c: lkji kji ji i
        // d: ponm onm nm m

        a = _mm_add_ps(a, presum); // sdcba scba sba sa

        sa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)); // sdcba{4}
        sc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 3, 3, 3)); // lkij{4}

        b = _mm_add_ps(b, sa); // hgfe gfe fe e + sdcba{4}
        d = _mm_add_ps(d, sc); // ponm onm nm m + lkij{4}

        sb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)); // shgfedcba{4}

        c = _mm_add_ps(c, sb); // lkji kji ji i + shgfedcba{4}
        d = _mm_add_ps(d, sb); // ponm onm nm m + lkij{4} + shgfedcba{4}

        presum = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 3, 3, 3)); // sa-p {4}

        // clang is smart enough to not do an insert and keeps us from having to fiddle with addresses
        x0 = _mm256_setr_m128(a, b);
        x1 = _mm256_setr_m128(c, d);

        dstv[i+0] = x0;
        dstv[i+1] = x1;
    }
}

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.4f ", xs[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    float v = 0.1;
    float temp = 0.1;
    if (argc >= 2) { sscanf(argv[1], "%f", &v); }
    if (argc >= 3) { sscanf(argv[1], "%f", &temp); }
    printf("init=%.2f temp=%.2f\n", v, temp);

    size_t N = 256;
    float* src = aligned_alloc(32, sizeof(float)*N);
    float* dst = aligned_alloc(32, sizeof(float)*N);

    for (size_t i = 0; i < N; i++) {
        src[i] = v;
    }

    for (size_t i = 0; i < 10000000; i++) {
        softmax_temp(src, dst, N, 1, temp);
    }

    dump_array(src, N);
    dump_array(dst, N);
}
