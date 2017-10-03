// +build linux,cgo,!arm64

package cupti

// #include <cupti.h>
import "C"
import (
	"fmt"

	"github.com/pkg/errors"
)

type VersionInfo struct {
	Version int
}

func (v VersionInfo) String() string {
	return fmt.Sprintf("CUPTI Version = %v", v.Version)
}

func Version() (VersionInfo, error) {
	var version C.uint32_t
	err := checkCUPTIError(C.cuptiGetVersion(&version))
	if err != nil {
		return VersionInfo{}, errors.Errorf("got error code = %d while calling cuptiGetVersion", err)
	}
	return VersionInfo{Version: int(version)}, nil
}
