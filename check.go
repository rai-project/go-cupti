package cupti

// #include <cupti.h>
import "C"
import (
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
)

func checkError(code types.CUptiResult) error {
	if code == types.CUPTI_SUCCESS {
		return nil
	}
	var errstr *C.char
	C.cuptiGetResultString(C.CUptiResult(code), &errstr)
	return errors.Errorf("cupti code = %s, error = %s", code.String(), C.GoString(errstr))
}
