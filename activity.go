package cupti

// #cgo CFLAGS: -I . -I /usr/local/cuda/include -I /usr/local/cuda/extras/CUPTI/include -DFMT_HEADER_ONLY
// #cgo LDFLAGS: -L . -lcupti -lcudart
// #cgo amd64 LDFLAGS: -L /usr/local/cuda/lib64 -L /usr/local/cuda/extras/CUPTI/lib64
// #cgo ppc64le LDFLAGS: -L /usr/local/cuda/lib -L /usr/local/cuda/extras/CUPTI/lib
// #include <cupti.h>
import "C"
import "github.com/rai-project/go-cupti/types"

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
	default:
		break
	}
	return kind.String()
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

func GetActivityObjectKindId(kind types.CUpti_ActivityObjectKind, id *C.CUpti_ActivityObjectKindId) uint {
	switch kind {
	// case types.CUPTI_ACTIVITY_OBJECT_PROCESS:
	// return (*id).__pt
	// 	return id.pt.processId
	// return (*id).pt.processId
	// case types.CUPTI_ACTIVITY_OBJECT_THREAD:
	// 	return (*id).pt.threadId
	// case types.CUPTI_ACTIVITY_OBJECT_DEVICE:
	// 	return (*id).dcs.deviceId
	// case types.CUPTI_ACTIVITY_OBJECT_CONTEXT:
	// 	return (*id).dcs.contextId
	// case types.CUPTI_ACTIVITY_OBJECT_STREAM:
	// 	return (*id).dcs.streamId
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

func cuptiActivityFlushAll() {

}

func cuptiActivityGetNextRecord() {

}

func cuptiActivityGetNumDroppedRecords() {

}

func cuptiActivityRegisterCallbacks() {

}
