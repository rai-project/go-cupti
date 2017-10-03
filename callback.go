// +build linux,cgo,!arm64

package cupti

/*
#include <cupti.h>
*/
import "C"
import (
	"unsafe"

	humanize "github.com/dustin/go-humanize"
	"github.com/ianlancetaylor/demangle"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	context "golang.org/x/net/context"
)

const (
	BUFFER_SIZE = 32 * 16384
	ALIGN_SIZE  = 8
)

func demangleName(n *C.char) string {
	if n == nil {
		return ""
	}
	mangledName := C.GoString(n)
	name, err := demangle.ToString(mangledName)
	if err != nil {
		return mangledName
	}
	return name
}

func cuptiEnableCallback(subscriber C.CUpti_SubscriberHandle, domain C.CUpti_CallbackDomain, cbid C.CUpti_CallbackId) error {
	return checkCUPTIError(C.cuptiEnableCallback( /*enable=*/ 1, subscriber, domain, cbid))
}

func (c *CUPTI) addCallback(name string) error {
	if cbid, err := types.CUpti_driver_api_trace_cbidString(name); err == nil {
		return cuptiEnableCallback(c.subscriber, C.CUPTI_CB_DOMAIN_DRIVER_API, C.CUpti_CallbackId(cbid))
	}
	if cbid, err := types.CUPTI_RUNTIME_TRACE_CBIDString(name); err == nil {
		return cuptiEnableCallback(c.subscriber, C.CUPTI_CB_DOMAIN_RUNTIME_API, C.CUpti_CallbackId(cbid))
	}
	return errors.Errorf("cannot find callback %v by name", name)
}

//export bufferRequested
func bufferRequested(buffer **C.uint8_t, size *C.size_t,
	maxNumRecords *C.size_t) {
	*size = BUFFER_SIZE + ALIGN_SIZE
	*buffer = (*C.uint8_t)(C.calloc(1, *size))
	if *buffer == nil {
		panic("ran out of memory while performing bufferRequested")
	}
	*maxNumRecords = 0
}

//export bufferCompleted
func bufferCompleted(ctx C.CUcontext, streamId C.uint32_t, buffer *C.uint8_t,
	size C.size_t, validSize C.size_t) {

}

func (c *CUPTI) addCorrelationData(correlation_id uint32, name string) {

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

func (c *CUPTI) onCudaConfigureCallEnter(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cudaConfigureCall_v3020_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"grid_dim":          []int{int(params.gridDim.x), int(params.gridDim.y), int(params.gridDim.z)},
		"block_dim":         []int{int(params.blockDim.x), int(params.blockDim.y), int(params.blockDim.z)},
		"shared_mem":        uint64(params.sharedMem),
		"shared_mem_human":  humanize.Bytes(uint64(params.sharedMem)),
		"stream":            uintptr(unsafe.Pointer(params.stream)),
	}
	if cbInfo.symbolName != nil {
		tags["symbol_name"] = C.GoString(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "configure_call", tags)
	if functionName != "" {
		ext.Component.Set(span, functionName)
	}
	c.ctx = setSpanContextCorrelationId(c.ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaConfigureCallExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
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

func (c *CUPTI) onCudaConfigureCall(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {

	switch cbInfo.callbackSite {
	case C.CUPTI_API_ENTER:
		return c.onCudaConfigureCallEnter(domain, cbid, cbInfo)
	case C.CUPTI_API_EXIT:
		return c.onCudaConfigureCallExit(domain, cbid, cbInfo)
	default:
		return errors.New("invalid callback site " + types.CUpti_ApiCallbackSite(cbInfo.callbackSite).String())
	}

	return nil

}

func (c *CUPTI) onCULaunchKernelEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	params := (*C.cuLaunchKernel_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"stream":            uintptr(unsafe.Pointer(params.hStream)),
		"grid_dim":          []int{int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)},
		"block_dim":         []int{int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)},
		"shared_mem":        uint64(params.sharedMemBytes),
		"shared_mem_human":  humanize.Bytes(uint64(params.sharedMemBytes)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "launch_kernel", tags)
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
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "device_synchronize", tags)
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

func (c *CUPTI) onCudaMemCopyDeviceEnter(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	if _, err := spanFromContextCorrelationId(c.ctx, correlationId); err == nil {
		return errors.Errorf("span %d already exists", correlationId)
	}
	params := (*C.cuMemcpyHtoD_v2_params)(cbInfo.functionParams)
	functionName := demangleName(cbInfo.functionName)
	tags := opentracing.Tags{
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uintptr(params.ByteCount),
		"byte_count_human":  humanize.Bytes(uint64(params.ByteCount)),
		"destination_ptr":   uintptr(params.dstDevice),
		"source_ptr":        uintptr(unsafe.Pointer(params.srcHost)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "cuda_memcpy_dev", tags)
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
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "cuda_launch", tags)
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
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "cuda_synchronize", tags)
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
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     functionName,
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"byte_count":        uint64(params.count),
		"byte_count_human":  humanize.Bytes(uint64(params.count)),
		"destination_ptr":   uintptr(unsafe.Pointer(params.dst)),
		"source_ptr":        uintptr(unsafe.Pointer(params.src)),
		"kind":              types.CUDAMemcpyKind(params.kind).String(),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = demangleName(cbInfo.symbolName)
	}
	span, _ := c.tracer.StartSpanFromContext(c.ctx, "cuda_memcpy", tags)
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
				Debug("skipping runtime call")
			return
		}
	case types.CUPTI_CB_DOMAIN_RUNTIME_API:
		cbid := types.CUPTI_RUNTIME_TRACE_CBID(cbid0)
		switch cbid {
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020,
			types.CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			handle.onCudaDeviceSynchronize(domain, cbid, cbInfo)
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
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
			handle.onCudaConfigureCall(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
			handle.onCudaSetupArgument(domain, cbid, cbInfo)
			return
		default:
			log.WithField("cbid", cbid.String()).
				WithField("function_name", demangleName(cbInfo.functionName)).
				Debug("skipping runtime call")
			return
		}
	}
}
