#!/bin/sh

nvprof --metrics inst_fp_32,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,inst_executed,dram_read_transactions,dram_write_transactions \
  --csv --log-file output.csv \
  ./nocupti
