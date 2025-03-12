#include <stdio.h>
#include <immintrin.h>
#include <assert.h>

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
        x = _mm_add_ps(m128_scan(v[i]), sum);
        v[i] = x;
        sum = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

void scan_inplace_ss(float* xs, size_t n) {
    assert(n % 16 == 0);
    __m128* v = (__m128*)__builtin_assume_aligned(xs, 16);
    __m128 sum = _mm_set1_epi32(0);
    __m128 a, b, c, d, sa, sb, sc;
    for (size_t i = 0; i < n/4; i += 4) {
        a = m128_scan(v[i + 0]); // dcba cba ba a
        b = m128_scan(v[i + 1]); // hgfe gfe fe f
        c = m128_scan(v[i + 2]); // lkji
        d = m128_scan(v[i + 3]); // ponm
        sa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3));
        sb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3));
        sc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 3, 3, 3));

        sa = _mm_add_ps(sa, sum);
        sb = _mm_add_ps(sb, sa);
        sc = _mm_add_ps(sc, sb);

        b = _mm_add_ps(b, sa);
        c = _mm_add_ps(c, sb);
        d = _mm_add_ps(d, sc);

        sum = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 3, 3, 3));

        v[i + 0] = a;
        v[i + 1] = b;
        v[i + 2] = c;
        v[i + 3] = d;
    }
}

void dump_array(float* xs, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.3f ", xs[i]);
    }
    printf("\n");
}

int main() {

    size_t n = 16;
    float *a = aligned_alloc(16, n * sizeof(float));
    float *b = aligned_alloc(16, n * sizeof(float));
    float *c = aligned_alloc(16, n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        a[i] = 0.1 * i;
        b[i] = 0.1 * i;
        c[i] = 0.1 * i;
    }

    dump_array(b, n);
    scan_simple(b, n);
    dump_array(b, n);

    puts("");

    dump_array(a, n);
    scan_inplace(a, n);
    dump_array(a, n);

    puts("");

    dump_array(c, n);
    scan_inplace_ss(c, n);
    dump_array(c, n);

    free(a);
    free(b);
    free(c);
}
