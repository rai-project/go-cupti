#!/bin/sh

nvprof --events inst_executed,inst_executed_fma_pipe_s0,inst_executed_fma_pipe_s1,inst_executed_fma_pipe_s2,inst_executed_fma_pipe_s3,not_predicated_off_thread_inst_executed,thread_inst_executed,l2_subp0_write_sector_misses,l2_subp1_write_sector_misses,l2_subp0_read_sector_misses,l2_subp1_read_sector_misses \
  --metrics inst_fp_32,flop_count_sp,dram_read_transactions,dram_write_transactions \
  --csv --log-file output.csv \
  ./nocupti
