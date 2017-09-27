#ifndef __CHECK_HPP__
#define __CHECK_HPP__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cupti.h>

#include "cbits/fmt/format.h"

extern void log_debug(const char *msg);
extern void log_info(const char *msg);
extern void log_error(const char *msg);

static inline void gpuAssert(cudaError_t code, const char *file,
                             const char *func, int line) {
  if (code != cudaSuccess) {
    fmt::MemoryWriter msg;
    msg.write("CUDA_CHECK: {} in {} {} {}\n", cudaGetErrorString(code), func,
              file, line);
    log_error(msg.c_str());
  }
}

static inline void cuptiAssert(CUptiResult code, const char *file,
                               const char *func, int line, bool abort = false) {
  if (code != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(code, &errstr);
    fmt::MemoryWriter msg;
    msg.write("CUPI_CHECK: {} in {} {} {}\n", errstr, func, file, line);
    log_error(msg.c_str());
  }
}

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __func__, __LINE__); }
#define CUPTI_CHECK(ans)                                                       \
  { cuptiAssert((ans), __FILE__, __func__, __LINE__); }

#endif // __CHECK_HPP__
