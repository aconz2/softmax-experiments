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

[sleefreduxavx2.h](./sleefreduxavx2.h) contains an extracted version of simd 8xf32 expf (called `Sleef_finz_expf8_u10avx2` which is defined by [xexpf](https://github.com/shibatch/sleef/blob/8aaafe87231e22d2952cf5128aa6d1e1abda6d96/src/libm/sleefsimdsp.c#L2029)) from [sleef](https://github.com/shibatch/sleef) so that it could be inlined. This makes a small difference and the tests that say `sleef` use a linked version and `sleefredux` use an inlined version. BSL license applies to that file. [sleefreduxsse2.h](./sleefreduxsse2.h) is likewise for sse2. And same with [sleefreduxscalar.h](./sleefreduxscalar.h).

[presum.c](./presum.c) has some experiments testing 4 different prefix sum algorithms: scalar, simd, simd superscalar 2, simd superscalar 4. For length 256:

```
N=256
           scan_simple 0.72 ns/el 723.88 ms
          scan_inplace 0.46 ns/el 464.56 ms
      scan_inplace_ss2 0.46 ns/el 462.13 ms
      scan_inplace_ss4 0.39 ns/el 392.78 ms
```

Have since updated with some versions that handle unaligned to 16 byte addresses and lengths which are not multiples of 4, 8, or 16. I played with a few ways to handle the head and tail and like `scan_unaligned2` the best; most of them seem fairly the same in perf but the generated code is pretty nice for that one.

[softmax.c](./softmax.c) has quite a few variations that use 1) libm expf, sleef, or sleefredux; 2) prefix sum, prefix sum superscalar 4, or no prefix sum (`sum` means regular softmax, `presum` is a cdf) 3) temperature scaling (`tempdiv`) or not. There is another temperature variation testing mul by inverse or div but the compiler does that for us anyways so I commented it out. sse2 is much slower.

Interestingly, the running sum version (one pass for exp and sum, one pass for division) is only faster at N=16 and is then consistently slower than doing one pass for exp, another for sum, and a final for division. At larger than cache sizes I would expect this to change. It looks like this is because the main loop for exp in the running sum version is doing 8 vbroadcastss (!) whereas the three pass version has those hoisted. Turns out exp takes a lot of constants. Things look better by disabling loop unrolling (and has no effect for the 3pass except reducing code size) because it was unrolling 2 and running out of registers I think; exp uses 11 constants so when it tries to do ss2 it has to reload constants.

Added the scalar implementation of expf from sleef; just writing plain loops vectorizes quite well and it would handle unaligned and non mod8 sized vectors easily. Still not as fast as the avx2, it is still vbroadcastss in the loop and uses slightly different strategy for the conditional handling which loads an op from memory.

Interestingly, sleef uses the range check < -104 > 104 in the scalar version for 0 and inf, but < -104 > 100 in the sse2/avx2 version; though values seem to go to those before anyhow: 88.722839 and -103.972084. I know there's a ton I don't understand about floats (and I've been testing with ffast-math) so maybe those checks make it play nice there. But it makes me think if you know you're going to clip your values in the input, you can get rid of those. Yeah things are slower without ffast-math.

Added an nzni for nonzero nonifinity that removes the range check in the avx2 version so we promise to pass it floats in the rough range -100 to 88. It does reduce register count because two less constants.

```
(all these names have the `softmax_` prefix removed)
N=256
                        math_sum 2.35 ns/el 601.85 ms
                       sleef_sum 0.53 ns/el 134.84 ms
           sleefredux_scalar_sum 0.42 ns/el 107.66 ms
   sleefredux_scalar_running_sum 0.40 ns/el 102.70 ms
             sleefredux_sse2_sum 0.67 ns/el 170.68 ms
             sleefredux_avx2_sum 0.35 ns/el 88.54 ms
     sleefredux_avx2_running_sum 0.36 ns/el 91.60 ms
        sleefredux_avx2_nzni_sum 0.32 ns/el 82.25 ms
  sleefredux_avx2_nzni_running_sum 0.32 ns/el 82.27 ms
                     math_presum 3.15 ns/el 805.68 ms
                    sleef_presum 1.33 ns/el 339.31 ms
          sleefredux_avx2_presum 1.14 ns/el 292.24 ms
      sleefredux_avx2_presum_ss4 0.62 ns/el 157.90 ms
        sleefredux_scalar_sum256 0.42 ns/el 107.54 ms
                math_sum_tempdiv 2.37 ns/el 605.51 ms
               sleef_sum_tempdiv 0.57 ns/el 146.41 ms
     sleefredux_avx2_sum_tempdiv 0.39 ns/el 100.31 ms
             math_presum_tempdiv 3.16 ns/el 808.14 ms
            sleef_presum_tempdiv 1.35 ns/el 345.13 ms
  sleefredux_avx2_presum_tempdiv 1.18 ns/el 301.10 ms
  sleefredux_avx2_presum_ss4_tempdiv 0.65 ns/el 166.84 ms
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

Tested these with isolcpus=30-31 and didn't see much difference at a glance. I wish I knew how to use cgroupv2 cpuset isolation to make that easier.


Oh I see now you can build sleef with inline headers:

```
mkdir -p build install
cd sleef
cmake -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=install/ -S . -B build
cmake --build build -j --clean-first
cmake --install build
```
