package cupti

// #include <cuda_runtime_api.h>
import "C"
import (
	"time"
)

func (ti *CUPTI) DeviceReset() (time.Time, error) {
	now := time.Now()
	err := checkCUDAError(C.cudaDeviceReset())
	if err != nil {
		return time.Time{}, err
	}
	ti.deviceResetTime = now
	return now, nil
}
