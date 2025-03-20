#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "sleefreduxavx2.h"
#include "sleefreduxsse2.h"
#include "sleefreduxscalar.h"

int main() {
    float cur = 88.7;
    float inf = __builtin_inf();
    float ninf = -inf;
    while (cur != HUGE_VAL) {
        float a = expf(cur);
        float b = Sleef_redux_finz_expf_u10scalar(cur);
        float c = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(cur)));
        float d = _mm_cvtss_f32(Sleef_redux_finz_expf4_u10sse2(_mm_set1_ps(cur)));
        printf("%f = %f %f %f %f\n", cur, a, b, c, d);
        cur = nextafterf(cur, inf);
        if (isinf(a) && isinf(b) && isinf(c) && isinf(d)) {
            break;
        }
    }

    cur = -103.9;
    while (cur != -HUGE_VAL) {
        float a = expf(cur);
        float b = Sleef_redux_finz_expf_u10scalar(cur);
        float c = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(cur)));
        float d = _mm_cvtss_f32(Sleef_redux_finz_expf4_u10sse2(_mm_set1_ps(cur)));
        /*printf("%f = %f %f %f %f\n", cur, a, b, c, d);*/
        printf("%f = %a %a %a %a\n", cur, a, b, c, d);
        cur = nextafterf(cur, ninf);
        if (a == 0.0 && b == 0.0 && c == 0.0 && d == 0.0) {
            break;
        }
        cur = nextafterf(cur, -inf);
    }
}
