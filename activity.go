package cupti

// #include <cupti.h>
import "C"
import "github.com/rai-project/go-cupti/types"

func cuptiActivityEnable(kind types.CUpti_ActivityKind) error {
  e := C.cuptiActivityEnable(C.CUpti_ActivityKind(kind))
  return checkCUPTIError(e)
}

func cuptiActivityDisable(kind types.CUpti_ActivityKind) error {
  e := C.cuptiActivityDisable(C.CUpti_ActivityKind(kind))
  return checkCUPTIError(e)
}

func cuptiActivityFlushAll() {

}

func cuptiActivityGetNextRecord() {

}

func cuptiActivityGetNumDroppedRecords() {

}

func cuptiActivityRegisterCallbacks() {

}
