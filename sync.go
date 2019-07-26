// +build linux,cgo,!nogpu

package cupti

// #include <cuda_runtime.h>
import "C"

func synchronize() error {
	return checkCUDAError(C.cudaDeviceSynchronize())
}

func (c *CUPTI) Wait() {
	//synchronize()
	err := cuptiActivityFlushAll()
	if err != nil && err != (*Error)(nil) {
		log.WithError(err).Error("failed to flush all activities")
	}
}
