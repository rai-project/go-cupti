// +build linux,cgo,!arm64

package cupti

// #include <cupti.h>
import "C"
import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
)

type Error struct {
	Code types.CUptiResult
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	var errstr *C.char
	C.cuptiGetResultString(C.CUptiResult(e.Code), &errstr)
	return fmt.Sprintf("cupti error code = %s, message = %s", e.Code.String(), C.GoString(errstr))
}

func checkCUPTIError(code C.CUptiResult) error {
	if code == C.CUPTI_SUCCESS {
		return nil
	}
	return &Error{Code: types.CUptiResult(code)}
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
