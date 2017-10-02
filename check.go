// +build linux,cgo

package cupti

// #include <cupti.h>
import "C"
import (
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
)

func checkCUPTIError(code C.CUptiResult) error {
	if code == C.CUPTI_SUCCESS {
		return nil
	}
	var errstr *C.char
	C.cuptiGetResultString(code, &errstr)
	return errors.Errorf("cupti error code = %s, message = %s", types.CUptiResult(code).String(), C.GoString(errstr))
}

func checkCUResult(code C.CUresult) error {
	if code == C.CUDA_SUCCESS {
		return nil
	}
	return errors.Errorf("cuda error code = %s", types.CUresult(code).String())
}

func checkCUDAError(code C.cudaError_t) error {
	if code == C.cudaSuccess {
		return nil
	}
	errstr := C.cudaGetErrorString(code)
	return errors.Errorf("cuda error code = %s, message = %s", types.CUDAError(code).String(), C.GoString(errstr))
}
