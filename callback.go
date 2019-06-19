// +build linux,cgo,!arm64,!nogpu

package cupti

/*
#include <cupti.h>
#include <nvToolsExt.h>
#include <nvToolsExtSync.h>
#include <generated_nvtx_meta.h>
*/
import "C"
import (
	"bytes"
	"context"
	"encoding/binary"
	"time"
	"unsafe"

	//humanize "github.com/dustin/go-humanize"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	tracer "github.com/rai-project/tracer"
)

// demangling names adds overhead
func demangleName(n *C.char) string {
	if n == nil {
		return ""
	}
	mangledName := C.GoString(n)
	return mangledName
	// name, err := demangle.ToString(mangledName)
	// if err != nil {
	// 	return mangledName
	// }
	// return name
}

// Returns a timestamp normalized to correspond with the start and end timestamps reported in the CUPTI activity records.
// The timestamp is reported in nanoseconds.
func cuptiGetTimestamp() (uint64, error) {
	var val C.uint64_t
	err := checkCUPTIError(C.cuptiGetTimestamp(&val))
	if err != nil {
		log.WithError(err).Error("failed to get cuptiGetTimestamp")
		return 0, err
	}
	return uint64(val), nil
}

// Returns a timestamp corresponds to the tracer begin time.
// The timestamp is reported in nanoseconds.
func (c *CUPTI) currentTimeStamp() time.Time {
	val, err := cuptiGetTimestamp()
	if err != nil {
		log.WithError(err).Error("failed to get currentTimeStamp")
		return time.Unix(0, 0)
	}
	ret := c.beginTime.Add(time.Duration(uint64(val)-c.startTimeStamp) * time.Nanosecond)
	return ret
}

func (c *CUPTI) enableCallback(name string) error {
	if cbid, err := types.CUpti_driver_api_trace_cbidString(name); err == nil {
		return checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_DRIVER_API, C.CUpti_CallbackId(cbid)))
	}
	if cbid, err := types.CUPTI_RUNTIME_TRACE_CBIDString(name); err == nil {
		return checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_RUNTIME_API, C.CUpti_CallbackId(cbid)))
	}
	if cbid, err := types.CUpti_nvtx_api_trace_cbidString(name); err == nil {
		return checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_NVTX, C.CUpti_CallbackId(cbid)))
	}
	return errors.Errorf("cannot find callback %v by name", name)
}

type spanCorrelation struct {
	correlationId uint
}

func setSpanContextCorrelationId(ctx context.Context, correlationId uint, span opentracing.Span) context.Context {
	key := spanCorrelation{correlationId: correlationId}
	return context.WithValue(ctx, key, span)
}

func spanFromContextCorrelationId(ctx context.Context, correlationId uint) (opentracing.Span, error) {
	key := spanCorrelation{correlationId: correlationId}
	span, ok := ctx.Value(key).(opentracing.Span)
	if !ok {
		return nil, errors.Errorf("span for correlationId=%v was not found", correlationId)
	}
	return span, nil
}

// func (c *CUPTI) onCudaConfigureCallEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
// 	correlationId := uint(cbInfo.correlationId)
// 	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
// 		return errors.Errorf("span %d already exists", correlationId)
// 	}
// 	params := (*C.cudaConfigureCall_v3020_params)(cbInfo.functionParams)
// 	functionName := demangleName(cbInfo.functionName)
// 	tags := opentracing.Tags{
// 		"trace_source":      "cupti",
// 		"cupti_type":        "callback",
// 		"context_uid":       uint32(cbInfo.contextUid),
// 		"correlation_id":    correlationId,
// 		"function_name":     functionName,
// 		"cupti_domain":      domain.String(),
// 		"cupti_callback_id": cbid.String(),
// 		"grid_dim":          []int{int(params.gridDim.x), int(params.gridDim.y), int(params.gridDim.z)},
// 		"block_dim":         []int{int(params.blockDim.x), int(params.blockDim.y), int(params.blockDim.z)},
// 		"shared_mem":        uint64(params.sharedMem),
// 		// "shared_mem_human":  humanize.Bytes(uint64(params.sharedMem)),
// 		"stream": uintptr(unsafe.Pointer(params.stream)),
// 	}
// 	if cbInfo.symbolName != nil {
// 		tags["symbol_name"] = C.GoString(cbInfo.symbolName)
// 	}
// 	span, _ := tracer.StartSpanFromContext(
// 		c.ctx,
// 		tracer.SYSTEM_LIBRARY_TRACE,
// 		"configure_call",
// 		opentracing.StartTime(c.currentTimeStamp()),
// 		tags,
// 	)
// 	if functionName != "" {
// 		ext.Component.Set(span, functionName)
// 	}
// 	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

// 	return nil
// }

// func (c *CUPTI) onCudaConfigureCallExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
// 	correlationId := uint(cbInfo.correlationId)
// 	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
// 	if err != nil {
// 		return err
// 	}
// 	if span == nil {
// 		return errors.New("no span found")
// 	}
// 	if cbInfo.functionReturnValue != nil {
// 		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
// 		span.SetTag("result", types.CUresult(*cuError).String())
// 	}
// 	span.FinishWithOptions(opentracing.FinishOptions{
// 		FinishTime: c.currentTimeStamp(),
// 	})
// 	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)
// 	return nil
// }

// func (c *CUPTI) onCudaConfigureCall(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
// 	switch cbInfo.callbackSite {
// 	case C.CUPTI_API_ENTER:
// 		return c.onCudaConfigureCallEnter(domain, cbid, cbInfo)
// 	case C.CUPTI_API_EXIT:
// 		return c.onCudaConfigureCallExit(domain, cbid, cbInfo)
// 	default:
// 		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
// 	}

// 	return nil
// }

func (c *CUPTI) onCULaunchKernelEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cuLaunchKernel_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"stream":            uintptr(unsafe.Pointer(params.hStream)),
		"grid_dim":          []int{int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)},
		"block_dim":         []int{int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)},
		"shared_mem":        uint64(params.sharedMemBytes),
		// "shared_mem_human":  humanize.Bytes(uint64(params.sharedMemBytes)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "launch_kernel", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCULaunchKernelExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}

	if span == nil {
		return errors.New("no span found")
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)
	return nil
}

func (c *CUPTI) onCULaunchKernel(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCULaunchKernelEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCULaunchKernelExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaDeviceSynchronizeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "device_synchronize", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaDeviceSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)
	return nil
}

func (c *CUPTI) onCudaDeviceSynchronize(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaDeviceSynchronizeEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaDeviceSynchronizeExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaStreamSynchronizeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "stream_synchronize", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaStreamSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)
	return nil
}

func (c *CUPTI) onCudaStreamSynchronize(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaStreamSynchronizeEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaStreamSynchronizeExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMallocEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaMalloc_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.size),
		// "byte_count_human":  humanize.Bytes(uint64(params.size)),
		"destination_ptr": uintptr(unsafe.Pointer(params.devPtr)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaMalloc", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMallocExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaMalloc(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMallocEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMallocExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMallocHostEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaMallocHost_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.size),
		// "byte_count_human":  humanize.Bytes(uint64(params.size)),
		"destination_ptr": uintptr(unsafe.Pointer(*params.ptr)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaMallocHost", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMallocHostExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaMallocHost(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMallocHostEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMallocHostExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaHostAllocEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaHostAlloc_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.size),
		// "byte_count_human":  humanize.Bytes(uint64(params.size)),
		"host_ptr": uintptr(unsafe.Pointer(*params.pHost)),
		"flags":    params.flags,
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaHostAlloc", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaHostAllocExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaHostAlloc(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaHostAllocEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaHostAllocExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMallocManagedEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaMallocManaged_v6000_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.size),
		// "byte_count_human":  humanize.Bytes(uint64(params.size)),
		"ptr":   uintptr(unsafe.Pointer(params.devPtr)),
		"flags": params.flags,
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaMallocManaged", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMallocManagedExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaMallocManaged(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMallocManagedEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMallocManagedExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMallocPitchEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaMallocPitch_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"ptr":               uintptr(unsafe.Pointer(params.devPtr)),
		"pitch":             uintptr(unsafe.Pointer(params.pitch)),
		"width":             params.width,
		"height":            params.height,
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaMallocPitch", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMallocPitchExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)
	return nil
}

func (c *CUPTI) onCudaMallocPitch(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMallocPitchEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMallocPitchExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaFreeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaFree_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"ptr":               uintptr(unsafe.Pointer(params.devPtr)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaFree", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaFreeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaFree(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaFreeEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaFreeExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaFreeHostEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaFree_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"ptr":               uintptr(unsafe.Pointer(params.devPtr)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaFreeHost", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaFreeHostExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaFreeHost(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaFreeHostEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaFreeHostExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMemCopyDeviceEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cuMemcpyHtoD_v2_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.ByteCount),
		// "byte_count_human":  humanize.Bytes(uint64(params.ByteCount)),
		"destination_ptr": uintptr(params.dstDevice),
		"source_ptr":      uintptr(unsafe.Pointer(params.srcHost)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cuda_memcpy_dev", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMemCopyDeviceExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaMemCopyDevice(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMemCopyDeviceEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMemCopyDeviceExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaSetupArgument(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	return nil
}

func (c *CUPTI) onCudaLaunchEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cuda_launch", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaLaunchExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaLaunch(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaLaunchEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaLaunchExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaSynchronizeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cuda_synchronize", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaSynchronize(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaSynchronizeEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaSynchronizeExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaMemCopyEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaMemcpy_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uint64(params.count),
		// "byte_count_human":  humanize.Bytes(uint64(params.count)),
		"destination_ptr": uintptr(unsafe.Pointer(params.dst)),
		"source_ptr":      uintptr(unsafe.Pointer(params.src)),
		"kind":            types.CUDAMemcpyKind(params.kind).String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cuda_memcpy", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaMemCopyExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaMemCopy(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaMemCopyEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaMemCopyExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaIpcGetEventHandleEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaIpcGetEventHandle_v4010_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":          "cupti",
		"cupti_type":            "callback",
		"context_uid":           uint32(cbInfo.contextUid),
		"correlation_id":        correlationId,
		"function_name":         functionName,
		"cupti_domain":          domain.String(),
		"cupti_callback_id":     cbid.String(),
		"cuda_ipc_event_handle": uintptr(unsafe.Pointer(params.handle)),
		"cuda_event":            uintptr(unsafe.Pointer(params.event)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaIpcGetEventHandle", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaIpcGetEventHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaIpcGetEventHandle(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaIpcGetEventHandleEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaIpcGetEventHandleExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaIpcOpenEventHandleEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaIpcOpenEventHandle_v4010_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":          "cupti",
		"cupti_type":            "callback",
		"context_uid":           uint32(cbInfo.contextUid),
		"correlation_id":        correlationId,
		"function_name":         functionName,
		"cupti_domain":          domain.String(),
		"cupti_callback_id":     cbid.String(),
		"cuda_ipc_event_handle": params.handle,
		"cuda_event":            uintptr(unsafe.Pointer(params.event)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaIpcOpenEventHandle", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaIpcOpenEventHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaIpcOpenEventHandle(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaIpcOpenEventHandleEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaIpcOpenEventHandleExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaIpcGetMemHandleEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaIpcGetMemHandle_v4010_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":        "cupti",
		"cupti_type":          "callback",
		"context_uid":         uint32(cbInfo.contextUid),
		"correlation_id":      correlationId,
		"function_name":       functionName,
		"cupti_domain":        domain.String(),
		"cupti_callback_id":   cbid.String(),
		"ptr":                 uintptr(unsafe.Pointer(params.devPtr)),
		"cuda_ipc_mem_handle": uintptr(unsafe.Pointer(params.handle)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaIpcGetMemHandle", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaIpcGetMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaIpcGetMemHandle(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaIpcGetMemHandleEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaIpcGetMemHandleExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaIpcOpenMemHandleEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaIpcOpenMemHandle_v4010_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":        "cupti",
		"cupti_type":          "callback",
		"context_uid":         uint32(cbInfo.contextUid),
		"correlation_id":      correlationId,
		"function_name":       functionName,
		"cupti_domain":        domain.String(),
		"cupti_callback_id":   cbid.String(),
		"ptr":                 uintptr(unsafe.Pointer(params.devPtr)),
		"cuda_ipc_mem_handle": params.handle,
		"flags":               params.flags,
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaIpcOpenMemHandle", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaIpcOpenMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaIpcOpenMemHandle(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaIpcOpenMemHandleEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaIpcOpenMemHandleExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaIpcCloseMemHandleEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cudaIpcCloseMemHandle_v4010_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"ptr":               uintptr(unsafe.Pointer(params.devPtr)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "cudaIpcCloseMemHandle", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaIpcCloseMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onCudaIpcCloseMemHandle(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaIpcCloseMemHandleEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaIpcCloseMemHandleExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onNvtxRangeStartAEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.nvtxRangeStartA_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"message":           C.GoString(params.message),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangeStartA", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangeStartAExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangeStartA(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangeStartAEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangeStartAExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func union_to_ascii_ptr(cbytes [8]byte) (result *C.char) {
	buf := bytes.NewBuffer(cbytes[:])
	var ptr uint64
	if err := binary.Read(buf, binary.LittleEndian, &ptr); err == nil {
		uptr := uintptr(ptr)
		return (*C.char)(unsafe.Pointer(uptr))
	}
	return nil
}

func (c *CUPTI) onNvtxRangeStartExEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.nvtxRangeStartEx_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"message":           C.GoString(union_to_ascii_ptr(params.eventAttrib.message)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangeStartEx", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangeStartExExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangeStartEx(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangeStartExEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangeStartExExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onNvtxRangeEndEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.nvtxRangeEnd_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"nvtx_range_id":     uint64(params.id),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangeEnd", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangeEndExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangeEnd(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangeEndEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangeEndExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onNvtxRangePushAEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.nvtxRangePushA_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"message":           C.GoString(params.message),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangePushA", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangePushAExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangePushA(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangePushAEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangePushAExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onNvtxRangePushExEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.nvtxRangePushEx_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"message":           C.GoString(union_to_ascii_ptr(params.eventAttrib.message)),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangePushEx", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangePushExExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangePushEx(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangePushExEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangePushExExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onNvtxRangePopEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"trace_source":      "cupti",
		"cupti_type":        "callback",
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	span, _ := tracer.StartSpanFromContext(c.ctx, tracer.SYSTEM_LIBRARY_TRACE, "nvtxRangePop", tags)
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onNvtxRangePopExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, nil)

	return nil
}

func (c *CUPTI) onNvtxRangePop(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onNvtxRangePopEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onNvtxRangePopExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

//export callback
func callback(userData unsafe.Pointer, domain0 C.CUpti_CallbackDomain, cbid0 C.CUpti_CallbackId, cbInfo *C.CUpti_CallbackData) {
	handle := (*CUPTI)(unsafe.Pointer(userData))
	if handle == nil {
		log.Debug("expecting a cupti handle, but got nil")
		return
	}
	domain := types.CUpti_CallbackDomain(domain0)
	switch domain {
	case types.CUPTI_CB_DOMAIN_DRIVER_API:
		cbid := types.CUpti_driver_api_trace_cbid(cbid0)
		switch cbid {
		case types.CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
			handle.onCULaunchKernel(domain, cbid, cbInfo)
			return
		case types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
			handle.onCudaMemCopyDevice(domain, cbid, cbInfo)
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping driver call")
			return
		}
	case types.CUPTI_CB_DOMAIN_RUNTIME_API:
		cbid := types.CUPTI_RUNTIME_TRACE_CBID(cbid0)
		switch cbid {
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
			handle.onCudaDeviceSynchronize(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			handle.onCudaStreamSynchronize(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
			handle.onCudaMalloc(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
			handle.onCudaMallocHost(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020:
			handle.onCudaHostAlloc(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
			handle.onCudaMallocManaged(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020:
			handle.onCudaMallocPitch(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
			handle.onCudaFree(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
			handle.onCudaFreeHost(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
			types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
			handle.onCudaMemCopy(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
			handle.onCudaLaunch(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020:
			handle.onCudaSynchronize(domain, cbid, cbInfo)
			return
		// case types.CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
		// 	handle.onCudaConfigureCall(domain, cbid, cbInfo)
		// return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
			handle.onCudaSetupArgument(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010:
			handle.onCudaIpcGetEventHandle(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010:
			handle.onCudaIpcOpenEventHandle(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010:
			handle.onCudaIpcGetMemHandle(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010:
			handle.onCudaIpcOpenMemHandle(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010:
			handle.onCudaIpcOpenMemHandle(domain, cbid, cbInfo)
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping runtime call")
			return
		}
	case types.CUPTI_CB_DOMAIN_NVTX:
		cbid := types.CUpti_nvtx_api_trace_cbid(cbid0)
		switch cbid {
		case types.CUPTI_CBID_NVTX_nvtxRangeStartA:
			handle.onNvtxRangeStartA(domain, cbid, cbInfo)
		case types.CUPTI_CBID_NVTX_nvtxRangeStartEx:
			handle.onNvtxRangeStartEx(domain, cbid, cbInfo)
		case types.CUPTI_CBID_NVTX_nvtxRangeEnd:
			handle.onNvtxRangeEnd(domain, cbid, cbInfo)
		case types.CUPTI_CBID_NVTX_nvtxRangePushA:
			handle.onNvtxRangePushA(domain, cbid, cbInfo)
		case types.CUPTI_CBID_NVTX_nvtxRangePushEx:
			handle.onNvtxRangePushEx(domain, cbid, cbInfo)
		case types.CUPTI_CBID_NVTX_nvtxRangePop:
			handle.onNvtxRangePop(domain, cbid, cbInfo)
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping nvtx marker")
			return
		}
	}
}
