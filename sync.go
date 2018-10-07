// +build linux,cgo,!arm64

package cupti

// #include <cuda_runtime.h>
import "C"

func synchronize() error {
	return checkCUDAError(C.cudaDeviceSynchronize())
}

func (c *CUPTI) Wait() {
	//synchronize()
	if err := cuptiActivityFlushAll(); err != nil {
		log.WithError(err).Error("failed to flush all activities")
	}
}
