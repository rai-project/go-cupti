// +build linux,cgo,!arm64

package cupti

/*
#include <cupti.h>
//
// A process object requires that we identify the process ID. A
// thread object requires that we identify both the process and
// thread ID.
   typedef struct {
    uint32_t processId;
    uint32_t threadId;
  } CUpti_ActivityObjectKindId_pt;

  // A device object requires that we identify the device ID. A
   // context object requires that we identify both the device and
   // context ID. A stream object requires that we identify device,
   // context, and stream ID.
   //
  typedef struct {
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
  } CUpti_ActivityObjectKindId_dcs;
*/
import "C"
import (
	"unsafe"

	"github.com/rai-project/go-cupti/types"
)

func getMemcpyKindString(kind types.CUpti_ActivityMemcpyKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
		return "HtoD"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
		return "DtoH"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
		return "HtoA"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
		return "AtoH"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
		return "AtoA"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
		return "AtoD"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
		return "DtoA"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
		return "DtoD"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
		return "HtoH"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
		return "PtoP"
	default:
		break
	}
	return "<unknown>  " + kind.String()
}

func getActivityOverheadKindString(kind types.CUpti_ActivityOverheadKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
		return "COMPILER"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
		return "BUFFER_FLUSH"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
		return "INSTRUMENTATION"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
		return "RESOURCE"
	default:
		break
	}

	return "<unknown> " + kind.String()
}

func getActivityObjectKindString(kind types.CUpti_ActivityObjectKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_OBJECT_PROCESS:
		return "PROCESS"
	case types.CUPTI_ACTIVITY_OBJECT_THREAD:
		return "THREAD"
	case types.CUPTI_ACTIVITY_OBJECT_DEVICE:
		return "DEVICE"
	case types.CUPTI_ACTIVITY_OBJECT_CONTEXT:
		return "CONTEXT"
	case types.CUPTI_ACTIVITY_OBJECT_STREAM:
		return "STREAM"
	default:
		break
	}

	return "<unknown> " + kind.String()
}

// Maps a MemoryKind enum to a const string.
func getMemoryKindString(kind types.CUpti_ActivityMemoryKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
		return "Unknown"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
		return "Pageable"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
		return "Pinned"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
		return "Device"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
		return "Array"
	default:
		break
	}
	return "<unknown> " + kind.String()
}

func getActivityObjectKindId(kind types.CUpti_ActivityObjectKind, id *C.CUpti_ActivityObjectKindId) uint {
	if id == nil {
		return 0
	}
	switch kind {
	case types.CUPTI_ACTIVITY_OBJECT_PROCESS:
		pt := (*C.CUpti_ActivityObjectKindId_pt)(unsafe.Pointer(id))
		return uint(pt.processId)
	case types.CUPTI_ACTIVITY_OBJECT_THREAD:
		pt := (*C.CUpti_ActivityObjectKindId_pt)(unsafe.Pointer(id))
		return uint(pt.threadId)
	case types.CUPTI_ACTIVITY_OBJECT_DEVICE:
		dcs := (*C.CUpti_ActivityObjectKindId_dcs)(unsafe.Pointer(id))
		return uint(dcs.deviceId)
	case types.CUPTI_ACTIVITY_OBJECT_CONTEXT:
		dcs := (*C.CUpti_ActivityObjectKindId_dcs)(unsafe.Pointer(id))
		return uint(dcs.contextId)
	case types.CUPTI_ACTIVITY_OBJECT_STREAM:
		dcs := (*C.CUpti_ActivityObjectKindId_dcs)(unsafe.Pointer(id))
		return uint(dcs.streamId)
	default:
		break
	}

	return 0xffffffff
}

func getComputeApiKindString(kind types.CUpti_ActivityComputeApiKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_COMPUTE_API_CUDA:
		return "CUDA"
	case types.CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
		return "CUDA_MPS"
	default:
		break
	}

	return "<unknown> " + kind.String()
}

func cuptiActivityEnable(kind types.CUpti_ActivityKind) error {
	e := C.cuptiActivityEnable(C.CUpti_ActivityKind(kind))
	return checkCUPTIError(e)
}

func cuptiActivityDisable(kind types.CUpti_ActivityKind) error {
	e := C.cuptiActivityDisable(C.CUpti_ActivityKind(kind))
	return checkCUPTIError(e)
}

func cuptiActivityConfigurePCSampling(ctx C.CUcontext, conf C.CUpti_ActivityPCSamplingConfig) error {
	e := C.cuptiActivityConfigurePCSampling(ctx, &conf)
	return checkCUPTIError(e)
}

func cuptiActivityFlushAll() error {
	return checkCUPTIError(C.cuptiActivityFlushAll(0))
}

func cuptiActivityGetNextRecord() {

}

func cuptiActivityGetNumDroppedRecords() {

}

func cuptiActivityRegisterCallbacks() {

}
