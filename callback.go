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
	"encoding/binary"
	"fmt"
	"time"
	"unsafe"

	"github.com/ianlancetaylor/demangle"
	"github.com/k0kubun/pp"
	"github.com/spf13/cast"

	//humanize "github.com/dustin/go-humanize"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	spanlog "github.com/opentracing/opentracing-go/log"
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
		err := checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_DRIVER_API, C.CUpti_CallbackId(cbid)))
		if err != nil {
			log.WithError(err).WithField("name", name).WithField("domain", types.CUPTI_CB_DOMAIN_DRIVER_API.String()).Error("cannot enable driver callback")
		}
		return err
	}
	if cbid, err := types.CUPTI_RUNTIME_TRACE_CBIDString(name); err == nil {
		err := checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_RUNTIME_API, C.CUpti_CallbackId(cbid)))
		if err != nil {
			log.WithError(err).WithField("name", name).WithField("domain", types.CUPTI_CB_DOMAIN_RUNTIME_API.String()).Error("cannot enable runtime callback")
		}
		return err
	}
	if cbid, err := types.CUpti_nvtx_api_trace_cbidString(name); err == nil {
		err := checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_NVTX, C.CUpti_CallbackId(cbid)))
		if err != nil {
			log.WithError(err).WithField("name", name).WithField("domain", types.CUPTI_CB_DOMAIN_NVTX.String()).Error("cannot enable nvtx callback")
		}
		return err
	}
	if cbid, err := types.CUpti_CallbackIdResourceString(name); err == nil {
		err := checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, c.subscriber, C.CUPTI_CB_DOMAIN_RESOURCE, C.CUpti_CallbackId(cbid)))
		if err != nil {
			log.WithError(err).WithField("name", name).WithField("domain", types.CUPTI_CB_DOMAIN_RESOURCE.String()).Error("cannot enable resource callback")
		}
		return err
	}
	log.WithField("name", name).Error("cannot enable callback")
	return errors.Errorf("cannot find callback %v by name", name)
}

type spanKey struct {
	correlationId uint64
	tag           string
}

func (c *CUPTI) setSpanContextCorrelationId(span opentracing.Span, correlationId uint64, tag string) {
	key := spanKey{
		correlationId: correlationId,
		tag:           tag,
	}
	// log.WithField("correlation_id", correlationId).WithField("tag", tag).Error("span added")
	c.spans.Store(key, span)
}

func (c *CUPTI) removeSpanContextByCorrelationId(correlationId uint64, tag string) {
	// log.WithField("correlation_id", correlationId).WithField("tag", tag).Error("span removed")
	key := spanKey{
		correlationId: correlationId,
		tag:           tag,
	}
	c.spans.Delete(key)
}

func (c *CUPTI) spanFromContextCorrelationId(correlationId uint64, tag string) (opentracing.Span, error) {
	key := spanKey{
		correlationId: correlationId,
		tag:           tag,
	}
	val, ok := c.spans.Load(key)
	if !ok {
		// log.WithField("correlation_id", correlationId).WithField("tag", tag).Error("span not found")
		return nil, errors.Errorf("span for correlationId=%v was not found", correlationId)
	}
	span, ok := val.(opentracing.Span)
	if !ok {
		// log.WithField("correlation_id", correlationId).WithField("tag", tag).Error("span not correct type")
		return nil, errors.Errorf("span for correlationId=%v was not found", correlationId)
	}
	return span, nil
}

// func (c *CUPTI) onCudaConfigureCallEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
// 	correlationId := uint64(cbInfo.correlationId)
// 	if _, err := c.spanFromContextCorrelationId(correlationId); err == nil {
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
// 	c.setSpanContextCorrelationId(span,"configure_call", correlationId)

// 	return nil
// }

// func (c *CUPTI) onCudaConfigureCallExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
// 	correlationId := uint64(cbInfo.correlationId)
// 	span, err := c.spanFromContextCorrelationId(correlationId)
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
// 	c.removeSpanContextByCorrelationId(correlationId)
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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "launch_kernel")

	return nil
}

func (c *CUPTI) onCULaunchKernelExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "launch_kernel")
	if err != nil {
		return err
	}

	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	c.removeSpanContextByCorrelationId(correlationId, "launch_kernel")
	return nil
}

func (c *CUPTI) onCULaunchKernel(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		res := c.onCULaunchKernelEnter(domain, cbid, cbInfo)
		return res
	case C.CUPTI_API_EXIT:
		return c.onCULaunchKernelExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaDeviceSynchronizeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "device_synchronize")

	return nil
}

func (c *CUPTI) onCudaDeviceSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "device_synchronize")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "device_synchronize")
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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "stream_synchronize")

	return nil
}

func (c *CUPTI) onCudaStreamSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "stream_synchronize")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "stream_synchronize")
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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaMalloc"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaMalloc")

	return nil
}

func (c *CUPTI) onCudaMallocExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaMalloc")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaMalloc")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocHost"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaMallocHost")

	return nil
}

func (c *CUPTI) onCudaMallocHostExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocHost")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaMallocHost")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaHostAlloc"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaHostAlloc")

	return nil
}

func (c *CUPTI) onCudaHostAllocExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaHostAlloc")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaHostAlloc")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocManaged"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaMallocManaged")

	return nil
}

func (c *CUPTI) onCudaMallocManagedExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocManaged")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaMallocManaged")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocPitch"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaMallocPitch")

	return nil
}

func (c *CUPTI) onCudaMallocPitchExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaMallocPitch")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaMallocPitch")
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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaFree"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaFree")

	return nil
}

func (c *CUPTI) onCudaFreeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaFree")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaFree")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cudaFreeHost"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaFreeHost")

	return nil
}

func (c *CUPTI) onCudaFreeHostExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaFreeHost")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaFreeHost")

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
	correlationId := uint64(cbInfo.correlationId)
	if _, err := c.spanFromContextCorrelationId(correlationId, "cuda_memcpy_dev"); err == nil {
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
	c.setSpanContextCorrelationId(span, correlationId, "cuda_memcpy_dev")

	return nil
}

func (c *CUPTI) onCudaMemCopyDeviceExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cuda_memcpy_dev")
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
	c.removeSpanContextByCorrelationId(correlationId, "cuda_memcpy_dev")

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
	correlationId := uint64(cbInfo.correlationId)
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

	c.setSpanContextCorrelationId(span, correlationId, "cuda_launch")

	// pp.Println("onCudaLaunchEnter correlationId = ", int(correlationId))

	return nil
}

func (c *CUPTI) onCudaLaunchExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	// pp.Println("onCudaLaunch Exit correlationId = ", int(correlationId))
	span, err := c.spanFromContextCorrelationId(correlationId, "cuda_launch")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	c.removeSpanContextByCorrelationId(correlationId, "cuda_launch")

	return nil
}

func (c *CUPTI) onCudaLaunchCaptureEventsEnter(domain types.CUpti_CallbackDomain, callbackName string, cbInfo *C.CUpti_CallbackData) error {

	eventData, err := c.findEventDataByCUCtxID(uint32(cbInfo.contextUid))
	if err != nil {
		err = c.onContextCreateAddEventGroup(domain, cbInfo.context)
		if err != nil {
			log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot add event data")
		}
		eventData, err = c.findEventDataByCUCtxID(uint32(cbInfo.contextUid))
		if err != nil {
			log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot find added event data")
			return err
		}
		eventData.destroyAfterKernelLaunch = true
	}

	if len(c.metrics) != 0 {
		metricData, err := c.findMetricDataByCUCtxID(uint32(cbInfo.contextUid))
		if err != nil {
			err = c.onContextCreateAddMetricGroup(domain, cbInfo.context)
			if err != nil {
				log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot add metric data")
			}
			metricData, err = c.findMetricDataByCUCtxID(uint32(cbInfo.contextUid))
			if err != nil {
				log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot find added metric data")
				return err
			}
			metricData.destroyAfterKernelLaunch = true
		}

		if metricData.eventGroupSets.numSets > 1 { // you have set the kernel to replay
			return nil
		}
	}

	mode, err := types.CUpti_EventCollectionModeString("CUPTI_EVENT_COLLECTION_MODE_KERNEL")
	if err != nil {
		return err
	}

	err = checkCUPTIError(C.cuptiSetEventCollectionMode(eventData.cuCtx, C.CUpti_EventCollectionMode(mode)))
	if err != nil {
		log.WithError(err).WithField("mode", mode.String()).Error("failed to cuptiSetEventCollectionMode")
		return err
	}

	err = checkCUPTIError(C.cuptiEventGroupEnable(eventData.eventGroup))
	if err != nil {
		log.WithError(err).WithField("mode", mode.String()).Error("failed to cuptiEventGroupEnable")
		return err
	}

	return nil
}

func (c *CUPTI) onCudaLaunchCaptureMetricsEnter(domain types.CUpti_CallbackDomain, callbackName string, cbInfo *C.CUpti_CallbackData) error {
	pp.Println("onCudaLaunchCaptureMetricsEnter")

	metricData, err := c.findMetricDataByCUCtxID(uint32(cbInfo.contextUid))
	if err != nil {
		err = c.onContextCreateAddMetricGroup(domain, cbInfo.context)
		if err != nil {
			log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot add metric data")
		}
		metricData, err = c.findMetricDataByCUCtxID(uint32(cbInfo.contextUid))
		if err != nil {
			log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot find added metric data")
			return err
		}
		metricData.destroyAfterKernelLaunch = true
	}

	eventGroupSetsPtr := metricData.eventGroupSets
	numSets := eventGroupSetsPtr.numSets

	if numSets <= 1 {
		mode, err := types.CUpti_EventCollectionModeString("CUPTI_EVENT_COLLECTION_MODE_KERNEL")
		if err != nil {
			return err
		}

		err = checkCUPTIError(C.cuptiSetEventCollectionMode(metricData.cuCtx, C.CUpti_EventCollectionMode(mode)))
		if err != nil {
			log.WithError(err).WithField("mode", mode.String()).Error("failed to cuptiSetEventCollectionMode")
			return err
		}
	}
	// C.cudaDeviceSynchronize()

	eventGroupSets := (*[1 << 28]C.CUpti_EventGroupSet)(unsafe.Pointer(eventGroupSetsPtr.sets))[:numSets:numSets]
	for ii, eventGroupSet := range eventGroupSets {
		err = checkCUPTIError(C.cuptiEventGroupSetEnable(&eventGroupSet))
		if err != nil {
			log.WithError(err).WithField("index", ii).Error("failed to cuptiEventGroupSetEnable")
			return err
		}
	}

	return nil
}

func (c *CUPTI) onCudaLaunchCaptureMetricsExit(domain types.CUpti_CallbackDomain, callbackName string, cbInfo *C.CUpti_CallbackData) error {
	// C.cudaDeviceSynchronize()

	metricData, err := c.findMetricDataByCUCtxID(uint32(cbInfo.contextUid))
	if err != nil {
		log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot find metric data in onCudaLaunchCaptureMetricsExit")
		return err
	}

	eventGroupSetsPtr := metricData.eventGroupSets
	metricIds := metricData.metricIds
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, callbackName)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("nil span found")
	}

	numSets := eventGroupSetsPtr.numSets
	eventGroupSets := (*[1 << 28]C.CUpti_EventGroupSet)(unsafe.Pointer(eventGroupSetsPtr.sets))[:numSets:numSets]
	for ii, eventGroupSet := range eventGroupSets {
		numEventGroups := int(eventGroupSet.numEventGroups)
		eventGroups := (*[1 << 28]C.CUpti_EventGroup)(unsafe.Pointer(eventGroupSet.eventGroups))[:numEventGroups:numEventGroups]
		for _, eventGroup := range eventGroups {
			var groupDomain C.CUpti_EventDomainID
			grouDomainSize := (C.size_t)(unsafe.Sizeof(groupDomain))
			err := checkCUPTIError(
				C.cuptiEventGroupGetAttribute(
					eventGroup,
					C.CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
					&grouDomainSize,
					unsafe.Pointer(&groupDomain),
				),
			)
			if err != nil {
				log.WithError(err).
					WithField("index", ii).
					WithField("attribute", types.CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID.String()).
					Error("failed to get cuptiEventGroupGetAttribute for group domain")
				return err
			}

			var numTotalInstances uint32
			numTotalInstancesSize := (C.size_t)(unsafe.Sizeof(numTotalInstances))
			err = checkCUPTIError(
				C.cuptiDeviceGetEventDomainAttribute(
					C.CUdevice(metricData.deviceId),
					groupDomain,
					C.CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
					&numTotalInstancesSize,
					unsafe.Pointer(&numTotalInstances),
				),
			)
			if err != nil {
				log.WithError(err).
					WithField("index", ii).
					WithField("attribute", types.CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT.String()).
					Error("failed to get cuptiEventGroupGetAttribute")
				return err
			}

			var numInstances uint32
			numInstancesSize := (C.size_t)(unsafe.Sizeof(numInstances))
			err = checkCUPTIError(
				C.cuptiEventGroupGetAttribute(
					eventGroup,
					C.CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
					&numInstancesSize,
					unsafe.Pointer(&numInstances),
				),
			)
			if err != nil {
				log.WithError(err).
					WithField("index", ii).
					WithField("attribute", types.CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT.String()).
					Error("failed to get cuptiEventGroupGetAttribute")
				return err
			}

			var numEvents uint32
			numEventsSize := (C.size_t)(unsafe.Sizeof(numEvents))
			err = checkCUPTIError(
				C.cuptiEventGroupGetAttribute(
					eventGroup,
					C.CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
					&numEventsSize,
					unsafe.Pointer(&numEvents),
				),
			)
			if err != nil {
				log.WithError(err).
					WithField("index", ii).
					WithField("attribute", types.CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS.String()).
					Error("failed to get cuptiEventGroupGetAttribute")
				return err
			}

			eventIds := make([]C.CUpti_EventID, numEvents)
			eventIdsSize := (C.size_t)(int(unsafe.Sizeof(eventIds[0])) * int(numEvents))
			err = checkCUPTIError(
				C.cuptiEventGroupGetAttribute(
					eventGroup,
					C.CUPTI_EVENT_GROUP_ATTR_EVENTS,
					&eventIdsSize,
					unsafe.Pointer(&eventIds[0]),
				),
			)
			if err != nil {
				log.WithError(err).
					WithField("index", ii).
					WithField("attribute", types.CUPTI_EVENT_GROUP_ATTR_EVENTS.String()).
					Error("failed to get cuptiEventGroupGetAttribute")
				return err
			}

			eventValueArray := make([]C.uint64_t, int(numEvents))
			for ii, eventId := range eventIds {
				values := make([]C.size_t, numInstances)
				valuesSize := (C.size_t)(int(unsafe.Sizeof(values[0])) * int(numInstances))
				err = checkCUPTIError(
					C.cuptiEventGroupReadEvent(
						eventGroup,
						C.CUPTI_EVENT_READ_FLAG_NONE,
						eventId,
						&valuesSize,
						&values[0],
					),
				)
				if err != nil {
					log.WithError(err).
						WithField("index", ii).
						WithField("event_id", int(eventId)).
						WithField("attribute", types.CUPTI_EVENT_READ_FLAG_NONE.String()).
						Error("failed to get cuptiEventGroupReadEvent")
					return err
				}

				accum := int64(0)
				for _, value := range values {
					accum += int64(value)
				}

				// normalize the event value to represent the total number of
				// domain instances on the device
				normalized := float64(accum) * float64(numTotalInstances) / float64(numInstances)

				eventValueArray[ii] = C.uint64_t(normalized)
			}

			for metricName, metricId := range metricIds {
				var metricValue C.CUpti_MetricValue
				err = checkCUPTIError(
					C.cuptiMetricGetValue(
						C.CUdevice(metricData.deviceId),
						metricId,
						(C.size_t)(uintptr(numEvents)*unsafe.Sizeof(eventIds[0])),
						&eventIds[0],
						(C.size_t)(uintptr(numEvents)*unsafe.Sizeof(eventValueArray[0])),
						&eventValueArray[0],
						0, // kernelDuration
						&metricValue,
					),
				)
				if err != nil {
					// this is not a hard error. a metric might not be able to
					// be computed using the current event group, but subsequent
					// values of the next event group might be able to compute the
					// metric.
					continue
				}

				var metricValueKind C.CUpti_MetricValueKind
				metricValueKindSize := (C.size_t)(unsafe.Sizeof(metricValueKind))
				err = checkCUPTIError(
					C.cuptiMetricGetAttribute(
						metricId,
						C.CUPTI_METRIC_ATTR_VALUE_KIND,
						&metricValueKindSize,
						unsafe.Pointer(&metricValueKind),
					),
				)
				if err != nil {
					log.WithError(err).
						WithField("index", ii).
						WithField("attribute", types.CUPTI_METRIC_ATTR_VALUE_KIND.String()).
						Error("failed to get cuptiMetricGetAttribute")
					return err
				}

				switch types.CUpti_MetricValueKind(metricValueKind) {
				case types.CUPTI_METRIC_VALUE_KIND_DOUBLE:
					span.LogFields(spanlog.Float64(metricName, *(*float64)(unsafe.Pointer(&metricValue))))
				case types.CUPTI_METRIC_VALUE_KIND_UINT64:
					span.LogFields(spanlog.Uint64(metricName, *(*uint64)(unsafe.Pointer(&metricValue))))
				case types.CUPTI_METRIC_VALUE_KIND_INT64:
					span.LogFields(spanlog.Int64(metricName, *(*int64)(unsafe.Pointer(&metricValue))))
				case types.CUPTI_METRIC_VALUE_KIND_PERCENT:
					span.LogFields(spanlog.Float64(metricName, *(*float64)(unsafe.Pointer(&metricValue))))
				case types.CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
					span.LogFields(spanlog.Uint64(metricName, *(*uint64)(unsafe.Pointer(&metricValue))))
				case types.CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
					utilization := *(*types.CUpti_MetricValueUtilizationLevel)(unsafe.Pointer(&metricValue))
					span.LogFields(spanlog.String(metricName, utilization.String()))
				default:
					log.WithError(err).
						WithField("index", ii).
						WithField("metric_value", types.CUpti_MetricValueKind(metricValueKind).String()).
						Error("failed to get cast metric value")
					return err
				}
			}
		}

		// for _, eventGroup := range eventGroups {
		// 	err = checkCUPTIError(C.cuptiEventGroupDisable(eventGroup))
		// 	if err != nil {
		// 		log.WithError(err).Error("failed to cuptiEventGroupDisable")
		// 		return err
		// 	}
		// }
	}

	for _, eventGroupSet := range eventGroupSets {
		err = checkCUPTIError(C.cuptiEventGroupSetDisable(&eventGroupSet))
		if err != nil {
			log.WithError(err).Error("failed to cuptiEventGroupDisable")
			return err
		}
	}

	return nil
}

func (c *CUPTI) onCudaLaunchCaptureEventsExit(domain types.CUpti_CallbackDomain, callbackName string, cbInfo *C.CUpti_CallbackData) error {
	eventData, err := c.findEventDataByCUCtxID(uint32(cbInfo.contextUid))
	if err != nil {
		log.WithError(err).WithField("context_id", uint32(cbInfo.contextUid)).Error("cannot find event data")
		return err
	}

	eventGroup := eventData.eventGroup
	eventIds := eventData.eventIds

	var numInstances C.size_t
	numInstancesByteCount := (C.size_t)(unsafe.Sizeof(numInstances))
	err = checkCUPTIError(
		C.cuptiEventGroupGetAttribute(
			eventGroup,
			C.CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
			&numInstancesByteCount,
			unsafe.Pointer(&numInstances),
		),
	)
	if err != nil {
		log.WithError(err).Error("failed to cuptiEventGroupGetAttribute")
		return err
	}

	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, callbackName)
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("nil span found")
	}

	for eventName, eventId := range eventIds {
		values := make([]C.size_t, int(numInstances))
		valuesByteCount := C.size_t(C.sizeof_size_t * numInstances)
		err = checkCUPTIError(
			C.cuptiEventGroupReadEvent(
				eventGroup,
				C.CUPTI_EVENT_READ_FLAG_NONE,
				eventId,
				&valuesByteCount,
				&values[0],
			))
		if err != nil {
			log.WithError(err).
				WithField("event_name", eventName).
				WithField("mode", types.CUPTI_EVENT_READ_FLAG_NONE.String()).
				Error("failed to cuptiEventGroupReadEvent")
			return err
		}
		eventVal := int64(0)
		for ii := 0; ii < int(numInstances); ii++ {
			eventVal += int64(values[ii])
		}

		// pp.Println(eventName, "  ", eventVal)
		span.LogFields(spanlog.Int64(eventName, eventVal))
	}

	err = checkCUPTIError(C.cuptiEventGroupDisable(eventGroup))
	if err != nil {
		log.WithError(err).Error("failed to cuptiEventGroupDisable")
		return err
	}

	return nil
}

func (c *CUPTI) onCudaLaunch(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		res := c.onCudaLaunchEnter(domain, cbid, cbInfo)
		if res == nil {
			if len(c.events) != 0 {
				c.onCudaLaunchCaptureEventsEnter(domain, "cuda_launch", cbInfo)
			}
			if len(c.metrics) != 0 {
				c.onCudaLaunchCaptureMetricsEnter(domain, "cuda_launch", cbInfo)
			}
		}
		return res
	case C.CUPTI_API_EXIT:
		if len(c.metrics) != 0 {
			err := c.onCudaLaunchCaptureMetricsExit(domain, "cuda_launch", cbInfo)
			if err != nil {
				log.WithError(err).Error("failed at exit metrics")
			}
		}
		if len(c.events) != 0 {
			err := c.onCudaLaunchCaptureEventsExit(domain, "cuda_launch", cbInfo)
			if err != nil {
				log.WithError(err).Error("failed at exit events")
			}
		}
		return c.onCudaLaunchExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onCudaSynchronizeEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cuda_synchronize")

	return nil
}

func (c *CUPTI) onCudaSynchronizeExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cuda_synchronize")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cuda_synchronize")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cuda_memcpy")

	return nil
}

func (c *CUPTI) onCudaMemCopyExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cuda_memcpy")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cuda_memcpy")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaIpcGetEventHandle")

	return nil
}

func (c *CUPTI) onCudaIpcGetEventHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaIpcGetEventHandle")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaIpcGetEventHandle")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaIpcOpenEventHandle")

	return nil
}

func (c *CUPTI) onCudaIpcOpenEventHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaIpcOpenEventHandle")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaIpcOpenEventHandle")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaIpcGetMemHandle")

	return nil
}

func (c *CUPTI) onCudaIpcGetMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaIpcGetMemHandle")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaIpcGetMemHandle")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaIpcOpenMemHandle")

	return nil
}

func (c *CUPTI) onCudaIpcOpenMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaIpcOpenMemHandle")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaIpcOpenMemHandle")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "cudaIpcCloseMemHandle")

	return nil
}

func (c *CUPTI) onCudaIpcCloseMemHandleExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "cudaIpcCloseMemHandle")
	if err != nil {
		return err
	}
	if cbInfo.functionReturnValue != nil {
		cuError := (*C.CUresult)(cbInfo.functionReturnValue)
		span.SetTag("result", types.CUresult(*cuError).String())
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "cudaIpcCloseMemHandle")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangeStartA")

	return nil
}

func (c *CUPTI) onNvtxRangeStartAExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangeStartA")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangeStartA")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangeStartEx")

	return nil
}

func (c *CUPTI) onNvtxRangeStartExExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangeStartEx")
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
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangeStartEx")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangeEnd")

	return nil
}

func (c *CUPTI) onNvtxRangeEndExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangeEnd")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangeEnd")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangePushA")

	return nil
}

func (c *CUPTI) onNvtxRangePushAExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangePushA")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangePushA")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangePushEx")

	return nil
}

func (c *CUPTI) onNvtxRangePushExExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangePushEx")
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
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangePushEx")

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
	correlationId := uint64(cbInfo.correlationId)
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
	c.setSpanContextCorrelationId(span, correlationId, "nvtxRangePop")

	return nil
}

func (c *CUPTI) onNvtxRangePopExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_nvtx_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint64(cbInfo.correlationId)
	span, err := c.spanFromContextCorrelationId(correlationId, "nvtxRangePop")
	if err != nil {
		return err
	}
	if span == nil {
		return errors.New("no span found")
	}
	span.Finish()
	c.removeSpanContextByCorrelationId(correlationId, "nvtxRangePop")

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

func (c *CUPTI) onCUCtxCreateEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	return c.onContextCreate(domain, cbInfo.context)
}

func (c *CUPTI) onCUCtxCreateExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	return c.onContextDestroy(domain, cbInfo.context)
}

func (c *CUPTI) onCUCtxCreate(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCUCtxCreateEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCUCtxCreateExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil
}

func (c *CUPTI) onContextCreate(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {

	if err := c.onContextCreateAddEventGroup(domain, cuCtx); err != nil {
		return err
	}
	if err := c.onContextCreateAddMetricGroup(domain, cuCtx); err != nil {
		return err
	}

	return nil
}

func (c *CUPTI) onContextCreateAddMetricGroup(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {

	if len(c.metrics) == 0 {
		return nil
	}

	// pp.Println("onContextCreateAddMetricGroup")

	deviceId := uint32(0)
	err := checkCUPTIError(C.cuptiGetDeviceId(cuCtx, (*C.uint32_t)(&deviceId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	ctxId := uint32(0)
	err = checkCUPTIError(C.cuptiGetContextId(cuCtx, (*C.uint32_t)(&ctxId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	err = c.addMetricGroup(cuCtx, uint32(ctxId), uint32(deviceId))
	if err != nil {
		return errors.Wrap(err, "cannot add metric group")
	}

	return nil
}

func (c *CUPTI) onContextCreateAddEventGroup(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {
	if len(c.events) == 0 {
		return nil
	}

	deviceId := uint32(0)
	err := checkCUPTIError(C.cuptiGetDeviceId(cuCtx, (*C.uint32_t)(&deviceId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	ctxId := uint32(0)
	err = checkCUPTIError(C.cuptiGetContextId(cuCtx, (*C.uint32_t)(&ctxId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	err = c.addEventGroup(cuCtx, uint32(ctxId), uint32(deviceId))
	if err != nil {
		return errors.Wrap(err, "cannot add event group")
	}

	return nil
}

func (c *CUPTI) onContextDestroy(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {
	if err := c.onContextDestroyMetricGroup(domain, cuCtx); err != nil {
		return err
	}

	if err := c.onContextDestroyEventGroup(domain, cuCtx); err != nil {
		return err
	}
	return nil
}

func (c *CUPTI) onContextDestroyMetricGroup(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {
	if len(c.metrics) == 0 {
		return nil
	}

	deviceId := uint32(0)
	err := checkCUPTIError(C.cuptiGetDeviceId(cuCtx, (*C.uint32_t)(&deviceId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	ctxId := uint32(0)
	err = checkCUPTIError(C.cuptiGetContextId(cuCtx, (*C.uint32_t)(&ctxId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	err = c.removeMetricGroup(cuCtx, uint32(ctxId), uint32(deviceId))
	if err != nil {
		return errors.Wrap(err, "cannot remove metric group")
	}

	return nil
}

func (c *CUPTI) onContextDestroyEventGroup(domain types.CUpti_CallbackDomain, cuCtx C.CUcontext) error {
	if len(c.events) == 0 {
		return nil
	}

	deviceId := uint32(0)
	err := checkCUPTIError(C.cuptiGetDeviceId(cuCtx, (*C.uint32_t)(&deviceId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	ctxId := uint32(0)
	err = checkCUPTIError(C.cuptiGetContextId(cuCtx, (*C.uint32_t)(&ctxId)))
	if err != nil {
		return errors.Wrap(err, "unable to get device id when creating resource context")
	}

	err = c.removeEventGroup(cuCtx, uint32(ctxId), uint32(deviceId))
	if err != nil {
		return errors.Wrap(err, "cannot remove event group")
	}

	return nil
}

func (c *CUPTI) onResourceContextCreated(domain types.CUpti_CallbackDomain, cbid types.CUpti_CallbackIdResource, cbInfo *C.CUpti_ResourceData) error {
	return c.onContextCreate(domain, cbInfo.context)
}

func (c *CUPTI) onResourceContextDestroyStarting(domain types.CUpti_CallbackDomain, cbid types.CUpti_CallbackIdResource, cbInfo *C.CUpti_ResourceData) error {
	return c.onContextDestroy(domain, cbInfo.context)
}

//export callback
func callback(userData unsafe.Pointer, domain0 C.CUpti_CallbackDomain, cbid0 C.CUpti_CallbackId, cbInfo0 unsafe.Pointer) {
	handle := (*CUPTI)(unsafe.Pointer(userData))
	if handle == nil {
		log.Debug("expecting a cupti handle, but got nil")
		return
	}
	domain := types.CUpti_CallbackDomain(domain0)
	switch domain {
	case types.CUPTI_CB_DOMAIN_DRIVER_API:
		cbid := types.CUpti_driver_api_trace_cbid(cbid0)
		cbInfo := (*C.CUpti_CallbackData)(cbInfo0)
		switch cbid {
		// case types.CUPTI_DRIVER_TRACE_CBID_cuCtxCreate, types.CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2:
		// 	handle.onCUCtxCreate(domain, cbid, cbInfo)
		// 	return
		case types.CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
			err := handle.onCULaunchKernel(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
			types.CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
			err := handle.onCudaMemCopyDevice(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping driver call")
			return
		}
	case types.CUPTI_CB_DOMAIN_RUNTIME_API:
		cbid := types.CUPTI_RUNTIME_TRACE_CBID(cbid0)
		cbInfo := (*C.CUpti_CallbackData)(cbInfo0)
		switch cbid {
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
			err := handle.onCudaDeviceSynchronize(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			err := handle.onCudaStreamSynchronize(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
			err := handle.onCudaMalloc(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
			err := handle.onCudaMallocHost(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020:
			err := handle.onCudaHostAlloc(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
			err := handle.onCudaMallocManaged(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020:
			err := handle.onCudaMallocPitch(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
			err := handle.onCudaFree(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
			err := handle.onCudaFreeHost(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
			types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
			err := handle.onCudaMemCopy(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020, types.CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
			err := handle.onCudaLaunch(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020:
			err := handle.onCudaSynchronize(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		// case types.CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
		// 	handle.onCudaConfigureCall(domain, cbid, cbInfo)
		// return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
			err := handle.onCudaSetupArgument(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010:
			err := handle.onCudaIpcGetEventHandle(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010:
			err := handle.onCudaIpcOpenEventHandle(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010:
			err := handle.onCudaIpcGetMemHandle(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010:
			err := handle.onCudaIpcOpenMemHandle(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010:
			err := handle.onCudaIpcOpenMemHandle(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping runtime call")
			return
		}
	case types.CUPTI_CB_DOMAIN_NVTX:
		cbid := types.CUpti_nvtx_api_trace_cbid(cbid0)
		cbInfo := (*C.CUpti_CallbackData)(cbInfo0)
		switch cbid {
		case types.CUPTI_CBID_NVTX_nvtxRangeStartA:
			err := handle.onNvtxRangeStartA(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_NVTX_nvtxRangeStartEx:
			err := handle.onNvtxRangeStartEx(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_NVTX_nvtxRangeEnd:
			err := handle.onNvtxRangeEnd(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_NVTX_nvtxRangePushA:
			err := handle.onNvtxRangePushA(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_NVTX_nvtxRangePushEx:
			err := handle.onNvtxRangePushEx(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_NVTX_nvtxRangePop:
			err := handle.onNvtxRangePop(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Info("skipping nvtx marker")
			return
		}
	case types.CUPTI_CB_DOMAIN_RESOURCE:
		cbid := types.CUpti_CallbackIdResource(cbid0)
		cbInfo := (*C.CUpti_ResourceData)(cbInfo0)
		switch cbid {
		case types.CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
			err := handle.onResourceContextCreated(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		case types.CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
			err := handle.onResourceContextDestroyStarting(domain, cbid, cbInfo)
			if err != nil {
				log.WithError(err).
					WithField("domain", domain.String()).
					WithField("callback", cbid.String()).Error("failed during callback")
			}
			return
		default:
			log.WithField("cbid", cbid.String()).
				Info("skipping resource domain event")
			return
		}
	}
}

//export kernelReplayCallback
func kernelReplayCallback(cKernelName *C.char, numReplaysDone C.int, customData unsafe.Pointer) {
	mangledName := C.GoString(cKernelName)
	kernelName, err := demangle.ToString(mangledName)
	if err != nil {
		kernelName = mangledName
	}
	fmt.Println(kernelName + " was called " + cast.ToString(int(numReplaysDone)) + " times")
}
