package cupti

// #include <cuda_runtime_api.h>
import "C"
import (
	"time"

	"github.com/rai-project/go-cupti/types"
)

func (ti *CUPTI) DeviceReset() (time.Time, error) {
	now := time.Now()
	errCode := C.cudaDeviceReset()
	if err := checkCUDAError(types.CUDAError(errCode)); err != nil {
		return time.Time{}, err
	}
	ti.deviceResetTime = now
	return now, nil
}
