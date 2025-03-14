#!/usr/bin/env bash

# symbols=softmax_math_sum,softmax_math_cumsum,softmax_sleef_sum,softmax_sleef_cumsum,softmax_sleefredux_sum,softmax_sleefredux_cumsum,softmax_math_sum_tempdiv,softmax_math_cumsum_tempdiv,softmax_sleef_sum_tempdiv,softmax_sleef_cumsum_tempdiv,softmax_sleefredux_sum_tempdiv,softmax_sleefredux_cumsum_tempdiv,softmax_math_sum_tempmul,softmax_math_cumsum_tempmul,softmax_sleef_sum_tempmul,softmax_sleef_cumsum_tempmul,softmax_sleefredux_sum_tempmul,softmax_sleefredux_cumsum_tempmul
symbols=softmax_sleefredux_presum_ss4_tempdiv

llvm-objdump --disassembler-color=terminal --x86-asm-syntax=intel --symbolize-operands --disassemble-symbols=$symbols softmax

llvm-objdump --disassembler-color=terminal --x86-asm-syntax=intel --symbolize-operands --disassemble-symbols=scan_inplace_ss4 presum

llvm-objdump --disassembler-color=terminal --x86-asm-syntax=intel --symbolize-operands --disassemble-symbols=binary_search,binary_search8 search
