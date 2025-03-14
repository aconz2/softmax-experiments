experiments in a fast softmax in c

`y = softmax(x)` is:

```
t = exp(x / T)
s = sum(t)
y = t / s
cdf = cumsum(y)
```

for an input `x` and temperature `T` (T could be 1 and known ahead of time and we can skip the division). And commonly if you are going to immediately sample from the distribution, you then take the cumulative sum.

All code assumes float32, avx2, and vector length at least 16 (with my primary focus 256). No precision analysis has been done (besides what sleef provides in exp). Tests are on AMD 5950x

[sleefredux.h](./sleefredux.h) contains an extracted version of simd 8xf32 expf (called `Sleef_finz_expf8_u10avx2` which is defined by [xexpf](https://github.com/shibatch/sleef/blob/8aaafe87231e22d2952cf5128aa6d1e1abda6d96/src/libm/sleefsimdsp.c#L2029)) from [sleef](https://github.com/shibatch/sleef) so that it could be inlined. This makes a small difference and the tests that say `sleef` use a linked version and `sleefredux` use an inlined version. BSL license applies to that file

[presum.c](./presum.c) has some experiments testing 4 different prefix sum algorithms: scalar, simd, simd superscalar 2, simd superscalar 4. For length 256:

```
N=256
           scan_simple 0.72 ns/el 723.88 ms
          scan_inplace 0.46 ns/el 464.56 ms
      scan_inplace_ss2 0.46 ns/el 462.13 ms
      scan_inplace_ss4 0.39 ns/el 392.78 ms
```

[softmax.c](./softmax.c) has quite a few variations that use 1) libm expf, sleef, or sleefredux; 2) prefix sum, prefix sum superscalar 4, or no prefix sum (`sum` means regular softmax, `presum` is a cdf) 3) temperature scaling (`tempdiv`) or not. There is another temperature variation testing mul by inverse or div but the compiler does that for us anyways so I commented it out.

```
(all these names have the `softmax_` prefix removed)
N=256
                        math_sum 2.23 ns/el 570.69 ms
                       sleef_sum 0.50 ns/el 127.34 ms
                  sleefredux_sum 0.33 ns/el 84.07 ms
                     math_presum 2.97 ns/el 760.55 ms
                    sleef_presum 1.24 ns/el 318.28 ms
               sleefredux_presum 1.07 ns/el 274.11 ms
           sleefredux_presum_ss4 0.58 ns/el 147.47 ms
                math_sum_tempdiv 2.23 ns/el 572.15 ms
               sleef_sum_tempdiv 0.52 ns/el 133.23 ms
          sleefredux_sum_tempdiv 0.39 ns/el 99.95 ms
             math_presum_tempdiv 2.98 ns/el 762.50 ms
            sleef_presum_tempdiv 1.27 ns/el 324.16 ms
       sleefredux_presum_tempdiv 1.15 ns/el 293.82 ms
   sleefredux_presum_ss4_tempdiv 0.64 ns/el 163.73 ms
```

The `scan_inplace_ss4` is copy pasted into softmax.c so it can do the temperature division all in one

[search.c](./search.c) has way too many variations of searching a sorted array, as if you were sampling from a cdf. fastest is `binary_search8` which is superscalar until the first element is found. There might be a bit of unfairness if the rng is faster because it gets called in batch, but idk. I tried higher number past 8 and it falls off.

```
RNG  0.77 ns/call 7.65 ms check=edcb38d7
N=256
                   linear_search 35.80 ns/call 128.88 ms check=1b603b46
                   binary_search 9.11 ns/call 32.81 ms check=1b603b46
                      xmm_search 19.39 ns/call 69.82 ms check=1b603b46
                      ymm_search 14.09 ns/call 50.71 ms check=1b603b46
                     ymm2_search 13.92 ns/call 50.10 ms check=1b603b46
                  binary_search2 8.70 ns/call 31.33 ms check=1b603b46
             binary_search2_easy 10.32 ns/call 37.16 ms check=1b603b46
             binary_search4_easy 9.77 ns/call 35.16 ms check=1b603b46
                  binary_search4 6.60 ns/call 23.75 ms check=1b603b46
             binary_search8_easy 9.43 ns/call 33.95 ms check=1b603b46
                  binary_search8 5.50 ns/call 19.79 ms check=1b603b46
                  binary_search9 5.44 ns/call 19.59 ms check=1b603b46
                 binary_search10 5.99 ns/call 21.56 ms check=1b603b46
                 binary_search12 5.95 ns/call 21.41 ms check=1b603b46
                 binary_search16 6.69 ns/call 24.07 ms check=1b603b46
                  ymm_search_256 14.32 ns/call 51.57 ms check=1b603b46
              binary_search8_256 5.49 ns/call 19.77 ms check=1b603b46
```

```
./run.sh
./dis.sh
```
