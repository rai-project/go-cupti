package cupti

// #include <cuda_runtime.h>
import "C"

func synchronize() error {
	return checkCUDAError(C.cudaDeviceSynchronize())
}

func (c *CUPTI) Wait() {
	synchronize()
}
