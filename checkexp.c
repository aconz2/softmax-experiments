#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "sleefreduxavx2.h"
#include "sleefreduxsse2.h"
#include "sleefreduxscalar.h"

/*void test_inf() {*/
/*    float cur = 88.7;*/
/*    float inf = __builtin_inf();*/
/*    while (cur != HUGE_VAL) {*/
/*        float a = expf(cur);*/
/*        float b = Sleef_redux_finz_expf_u10scalar(cur);*/
/*        float c = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        float d = _mm_cvtss_f32(Sleef_redux_finz_expf4_u10sse2(_mm_set1_ps(cur)));*/
/*        float e = _mm256_cvtss_f32(Sleef_redux_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        float f = Sleef_redux_finz_expf_u10scalar(cur);*/
/*        printf("%f = %f %f %f %f %f %f\n", cur, a, b, c, d, e, f);*/
/*        cur = nextafterf(cur, inf);*/
/*        if (isinf(a) && isinf(b) && isinf(c) && isinf(d) && isinf(e) && isinf(f)) {*/
/*            break;*/
/*        }*/
/*    }*/
/*}*/
/**/
/*void test_zero() {*/
/*    float cur = -103.9;*/
/*    float inf = __builtin_inf();*/
/*    float ninf = -__builtin_inf();*/
/*    while (cur != -HUGE_VAL) {*/
/*        float a = expf(cur);*/
/*        float b = Sleef_redux_finz_expf_u10scalar(cur);*/
/*        float c = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        float d = _mm_cvtss_f32(Sleef_redux_finz_expf4_u10sse2(_mm_set1_ps(cur)));*/
/*        float e = _mm256_cvtss_f32(Sleef_redux_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        float f = Sleef_redux_finz_expf_u10scalar(cur);*/
/*        printf("%f = %a %a %a %a %a %a\n", cur, a, b, c, d, e, f);*/
/*        cur = nextafterf(cur, ninf);*/
/*        if (a == 0.0 && b == 0.0 && c == 0.0 && d == 0.0 && e == 0.0 && f == 0.0) {*/
/*            break;*/
/*        }*/
/*        cur = nextafterf(cur, -inf);*/
/*    }*/
/*}*/
/**/
/*void test_finz() {*/
/*    float inf = __builtin_inf();*/
/*    float ninf = -inf;*/
/*    float cur = ninf;*/
/**/
/*    while (cur != HUGE_VAL) {*/
/*        float a = Sleef_redux_finz_expf_u10scalar(cur);*/
/*        float b = Sleef_redux_expf_u10scalar(cur);*/
/*        if (a == b) {*/
/*            cur = nextafterf(cur, inf);*/
/*        } else {*/
/*            float end = cur;*/
/*            float aa, bb;*/
/*            do {*/
/*                float prevend = end;*/
/*                end = nextafterf(end, inf);*/
/*                aa = Sleef_redux_finz_expf_u10scalar(end);*/
/*                bb = Sleef_redux_expf_u10scalar(end);*/
/*            } while (a == aa && end != HUGE_VAL);*/
/**/
/*            printf("%f : %f (%a : %a) a=%f (%a) b=%f (%a)\n", cur, end, cur, end, a, a, b, b);*/
/*            cur = end;*/
/**/
/*        }*/
/*    }*/
/*}*/
/**/
/*void test_finz_avx2() {*/
/*    float inf = __builtin_inf();*/
/*    float ninf = -inf;*/
/*    float cur = ninf;*/
/**/
/*    while (cur != HUGE_VAL) {*/
/*        float a = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        float b = _mm256_cvtss_f32(Sleef_redux_expf8_u10avx2(_mm256_set1_ps(cur)));*/
/*        if (a == b) {*/
/*            cur = nextafterf(cur, inf);*/
/*        } else {*/
/*            float end = cur;*/
/*            float aa, bb;*/
/*            do {*/
/*                float prevend = end;*/
/*                end = nextafterf(end, inf);*/
/*                aa = _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(end)));*/
/*                bb = _mm256_cvtss_f32(Sleef_redux_expf8_u10avx2(_mm256_set1_ps(end)));*/
/*            } while (a == aa && end != HUGE_VAL);*/
/**/
/*            printf("%f : %f (%a : %a) a=%f (%a) b=%f (%a)\n", cur, end, cur, end, a, a, b, b);*/
/*            cur = end;*/
/**/
/*        }*/
/*    }*/
/*}*/
/**/
typedef float (*Exp)(float);

float exp_scalar_finz(float x) { return Sleef_redux_finz_expf_u10scalar(x); }
float exp_scalar(float x) { return Sleef_redux_expf_u10scalar(x); }
float exp_sse2_finz(float x) { return _mm_cvtss_f32(Sleef_redux_finz_expf4_u10sse2(_mm_set1_ps(x))); }
float exp_sse2(float x) { return _mm_cvtss_f32(Sleef_redux_expf4_u10sse2(_mm_set1_ps(x))); }
float exp_avx2_finz(float x) { return _mm256_cvtss_f32(Sleef_redux_finz_expf8_u10avx2(_mm256_set1_ps(x))); }
float exp_avx2(float x) { return _mm256_cvtss_f32(Sleef_redux_expf8_u10avx2(_mm256_set1_ps(x))); }

void test_generic_away_from_zero(Exp f1, Exp f2) {
    float cur = 0;
    float a, b;

    do {
        cur = nextafterf(cur, -HUGE_VALF);
        a = f1(cur);
        b = f2(cur);
    } while (a == b && cur != -HUGE_VALF);
    printf("%f a=%f b=%f\n", cur, a, b);

    cur = 0;
    do {
        cur = nextafterf(cur, HUGE_VALF);
        a = f1(cur);
        b = f2(cur);
    } while (a == b && cur != HUGE_VALF);
    printf("%f a=%f b=%f\n", cur, a, b);
}


int main() {
    /*test_inf();*/
    /*test_zero();*/
    /*test_finz();*/
    /*test_finz_avx2();*/
    printf("scalar:\n");
    test_generic_away_from_zero(exp_scalar_finz, exp_scalar);
    printf("sse2:\n");
    test_generic_away_from_zero(exp_sse2_finz, exp_sse2);
    printf("avx2:\n");
    test_generic_away_from_zero(exp_avx2_finz, exp_avx2);
}
