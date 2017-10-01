package cupti

// #include <cupti.h>
import "C"
import (
	"github.com/pkg/errors"
)

func Version() (int, error) {
	var version C.uint32_t
	err := checkCUPTIError(C.cuptiGetVersion(&version))
	if err != nil {
		return 0, errors.Errorf("got error code = %d while calling cuptiGetVersion", err)
	}
	return int(version), nil
}
