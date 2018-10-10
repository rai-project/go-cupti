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

extern void bufferRequested(uint8_t ** buffer, size_t * size, size_t * maxNumRecords);
  extern void bufferCompleted(CUcontext ctx , uint32_t streamId , uint8_t * buffer ,
	size_t size , size_t validSize);
*/
import "C"
import (
	"time"
	"unsafe"

	humanize "github.com/dustin/go-humanize"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/rai-project/go-cupti/types"
)

func getActivityMemcpyKindString(kind types.CUpti_ActivityMemcpyKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
		return "h2d"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
		return "d2h"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
		return "h2a"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
		return "a2h"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
		return "a2a"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
		return "a2d"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
		return "d2a"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
		return "d2d"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
		return "h2h"
	case types.CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
		return "p2p"
	default:
		break
	}
	return "<unknown>  " + kind.String()
}

// Maps a MemoryKind enum to a const string.
func getActivityMemoryKindString(kind types.CUpti_ActivityMemoryKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
		return "unknown"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
		return "pageable"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
		return "pinned"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
		return "device"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
		return "array"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
		return "managed"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
		return "device_tatic"
	case types.CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
		return "managed_staic"
	default:
		break
	}
	return "<unknown> " + kind.String()
}

func getActivityOverheadKindString(kind types.CUpti_ActivityOverheadKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
		return "compiler"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
		return "buffer_flush"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
		return "instrumentation"
	case types.CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
		return "resource"
	default:
		break
	}

	return "<unknown> " + kind.String()
}

func getActivityObjectKindString(kind types.CUpti_ActivityObjectKind) string {
	switch kind {
	case types.CUPTI_ACTIVITY_OBJECT_PROCESS:
		return "process"
	case types.CUPTI_ACTIVITY_OBJECT_THREAD:
		return "thread"
	case types.CUPTI_ACTIVITY_OBJECT_DEVICE:
		return "device"
	case types.CUPTI_ACTIVITY_OBJECT_CONTEXT:
		return "context"
	case types.CUPTI_ACTIVITY_OBJECT_STREAM:
		return "stream"
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

// round x to the nearest multiple of y, larger or equal to x.
//
// from /usr/include/sys/param.h Macros for counting and rounding.
// #define roundup(x, y)   ((((x)+((y)-1))/(y))*(y))
//export roundup
func roundup(x, y C.size_t) C.size_t {
	return ((x + y - 1) / y) * y
}

//export bufferRequested
func bufferRequested(buffer **C.uint8_t, size *C.size_t,
	maxNumRecords *C.size_t) {
	*size = roundup(BUFFER_SIZE, ALIGN_SIZE)
	*buffer = (*C.uint8_t)(C.aligned_alloc(ALIGN_SIZE, *size))
	if *buffer == nil {
		panic("ran out of memory while performing bufferRequested")
	}
	*maxNumRecords = 0
}

//export bufferCompleted
func bufferCompleted(ctx C.CUcontext, streamId C.uint32_t, buffer *C.uint8_t,
	size C.size_t, validSize C.size_t) {

	if currentCUPTI == nil {
		log.Error("the current cupti instance is not found")
		return
	}
	currentCUPTI.activityBufferCompleted(ctx, streamId, buffer, size, validSize)
}

func (c *CUPTI) activityBufferCompleted(ctx C.CUcontext, streamId C.uint32_t, buffer *C.uint8_t,
	size C.size_t, validSize C.size_t) {
	defer func() {
		if buffer != nil {
			C.free(unsafe.Pointer(buffer))
		}
	}()
	if validSize <= 0 {
		return
	}
	var record *C.CUpti_Activity
	for {
		err0 := checkCUPTIError(C.cuptiActivityGetNextRecord(buffer, validSize, &record))
		if err0 == nil {
			if record == nil {
				break
			}
			c.processActivity(record)
			continue
		}
		err, ok := err0.(*Error)
		if !ok {
			panic("invalid error type")
		}
		if err.Code == types.CUPTI_ERROR_MAX_LIMIT_REACHED {
			break
		}

		log.WithError(err).Error("failed to get cupti cuptiActivityGetNextRecord")
	}

	var dropped C.size_t
	if err := checkCUPTIError(C.cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped)); err != nil {
		log.WithError(err).Error("failed to get cuptiDeviceGetTimestamp")
		return
	}
	if dropped != 0 {
		log.Infof("Dropped %v activity records", uint(dropped))
	}

}

func (c *CUPTI) processActivity(record *C.CUpti_Activity) {
	switch types.CUpti_ActivityKind(record.kind) {
	// https://docs.nvidia.com/cuda/cupti/index.html#structCUpti__ActivityMemcpy
	case types.CUPTI_ACTIVITY_KIND_MEMCPY:
		activity := (*C.CUpti_ActivityMemcpy)(unsafe.Pointer(record))
		startTime := c.beginTime.Add(time.Duration(uint64(activity.start)-c.startTimeStamp) * time.Nanosecond)
		endTime := c.beginTime.Add(time.Duration(uint64(activity.end)-c.startTimeStamp) * time.Nanosecond)
		sp, _ := opentracing.StartSpanFromContext(
			c.ctx,
			"gpu_memcpy",
			opentracing.StartTime(startTime),
			opentracing.Tags{
				"cupti_type":            "activity",
				"bytes":                 activity.bytes,
				"bytes_human":           humanize.Bytes(uint64(activity.bytes)),
				"copy_kind":             getActivityMemcpyKindString(types.CUpti_ActivityMemcpyKind(activity.copyKind)),
				"src_kind":              getActivityMemoryKindString(types.CUpti_ActivityMemoryKind(activity.srcKind)),
				"dst_kind":              getActivityMemoryKindString(types.CUpti_ActivityMemoryKind(activity.dstKind)),
				"device_id":             activity.deviceId,
				"context_id":            activity.contextId,
				"stream_id":             activity.streamId,
				"correlation_id":        activity.correlationId,
				"runtimeCorrelation_id": activity.runtimeCorrelationId,
			},
		)
		sp.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
	case types.CUPTI_ACTIVITY_KIND_MEMSET:
		activity := (*C.CUpti_ActivityMemset)(unsafe.Pointer(record))
		startTime := c.beginTime.Add(time.Duration(uint64(activity.start)-c.startTimeStamp) * time.Nanosecond)
		endTime := c.beginTime.Add(time.Duration(uint64(activity.end)-c.startTimeStamp) * time.Nanosecond)
		sp, _ := opentracing.StartSpanFromContext(
			c.ctx,
			"gpu_memset",
			opentracing.StartTime(startTime),
			opentracing.Tags{
				"cupti_type":     "activity",
				"bytes":          activity.bytes,
				"bytes_human":    humanize.Bytes(uint64(activity.bytes)),
				"memory_kind":    getActivityMemoryKindString(types.CUpti_ActivityMemoryKind(activity.memoryKind)),
				"value":          activity.value,
				"device_id":      activity.deviceId,
				"context_id":     activity.contextId,
				"stream_id":      activity.streamId,
				"correlation_id": activity.correlationId,
			},
		)
		sp.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
		// https://docs.nvidia.com/cuda/cupti/index.html#structCUpti__ActivityKernel4
	case types.CUPTI_ACTIVITY_KIND_KERNEL, types.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
		activity := (*C.CUpti_ActivityKernel4)(unsafe.Pointer(record))
		startTime := c.beginTime.Add(time.Duration(uint64(activity.start)-c.startTimeStamp) * time.Nanosecond)
		endTime := c.beginTime.Add(time.Duration(uint64(activity.end)-c.startTimeStamp) * time.Nanosecond)
		sp, _ := opentracing.StartSpanFromContext(
			c.ctx,
			"gpu_kernel",
			opentracing.StartTime(startTime),
			opentracing.Tags{
				"cupti_type":                 "activity",
				"name":                       activity.name,
				"grid_dim":                   []int{int(activity.gridX), int(activity.gridY), int(activity.gridZ)},
				"block_dim":                  []int{int(activity.blockX), int(activity.blockY), int(activity.blockZ)},
				"device_id":                  activity.deviceId,
				"context_id":                 activity.contextId,
				"stream_id":                  activity.streamId,
				"correlation_id":             activity.correlationId,
				"start":                      activity.start,
				"end":                        activity.completed,
				"queued":                     activity.queued,
				"submitted":                  activity.submitted,
				"local_mem":                  activity.localMemoryTotal,
				"dynamic_sharedMemory":       activity.dynamicSharedMemory,
				"dynamic_sharedMemory_human": humanize.Bytes(uint64(activity.dynamicSharedMemory)),
				"static_sharedMemory":        activity.staticSharedMemory,
				"static_sharedMemory_human":  humanize.Bytes(uint64(activity.staticSharedMemory)),
			},
		)
		sp.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
	case types.CUPTI_ACTIVITY_KIND_OVERHEAD:
		activity := (*C.CUpti_ActivityOverhead)(unsafe.Pointer(record))
		startTime := c.beginTime.Add(time.Duration(uint64(activity.start)-c.startTimeStamp) * time.Nanosecond)
		endTime := c.beginTime.Add(time.Duration(uint64(activity.end)-c.startTimeStamp) * time.Nanosecond)
		sp, _ := opentracing.StartSpanFromContext(
			c.ctx,
			"gpu_overhead",
			opentracing.StartTime(startTime),
			opentracing.Tags{
				"cupti_type":    "activity",
				"object_id":     activity.objectId,
				"object_kind":   getActivityObjectKindString(types.CUpti_ActivityObjectKind(activity.objectKind)),
				"overhead_kind": getActivityOverheadKindString(types.CUpti_ActivityOverheadKind(activity.overheadKind)),
			},
		)
		sp.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
	case types.CUPTI_ACTIVITY_KIND_DRIVER, types.CUPTI_ACTIVITY_KIND_RUNTIME:
		activity := (*C.CUpti_ActivityAPI)(unsafe.Pointer(record))
		startTime := c.beginTime.Add(time.Duration(uint64(activity.start)-c.startTimeStamp) * time.Nanosecond)
		endTime := c.beginTime.Add(time.Duration(uint64(activity.end)-c.startTimeStamp) * time.Nanosecond)
		sp, _ := opentracing.StartSpanFromContext(
			c.ctx,
			"gpu_api",
			opentracing.StartTime(startTime),
			opentracing.Tags{
				"cupti_type":     "activity",
				"cbid":           int(activity.cbid),
				"correlation_id": activity.correlationId,
				"kind":           activity.kind,
				"process_id":     activity.processId,
				"thread_id":      activity.threadId,
			},
		)
		sp.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
	default:
		log.Error("can not cast activity kind")
	}
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
	e := C.cuptiActivityFlushAll(0)
	return checkCUPTIError(e)
}

func cuptiActivityRegisterCallbacks() error {
	e := C.cuptiActivityRegisterCallbacks(
		(C.CUpti_BuffersCallbackRequestFunc)(unsafe.Pointer(C.bufferRequested)),
		(C.CUpti_BuffersCallbackCompleteFunc)(unsafe.Pointer(C.bufferCompleted)),
	)
	return checkCUPTIError(e)
}
