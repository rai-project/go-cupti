package cupti

// #include <cupti.h>
import "C"
import (
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
)

func checkCUPTIError(code types.CUptiResult) error {
	if code == types.CUPTI_SUCCESS {
		return nil
	}
	var errstr *C.char
	C.cuptiGetResultString(C.CUptiResult(code), &errstr)
	return errors.Errorf("cupti error code = %s, message = %s", code.String(), C.GoString(errstr))
}

func checkCUResult(code types.CUresult) error {
	if code == types.CUDA_SUCCESS {
		return nil
	}
	return errors.Errorf("cuda error code = %s", code.String())
}

func checkCUDAError(code types.CUDAError) error {
	if code == types.CUDASuccess {
		return nil
	}
	errstr := C.cudaGetErrorString(C.cudaError_t(code))
	return errors.Errorf("cuda error code = %s, message = %s", code.String(), C.GoString(errstr))
}
