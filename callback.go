package cupti

/*
#include <cupti.h>
*/
import "C"
import (
	"unsafe"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	context "golang.org/x/net/context"
)

const (
	BUFFER_SIZE = 32 * 16384
	ALIGN_SIZE  = 8
)

var (
	DefaultCallbacks = []string{
		"CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2",
	}
)

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
	tags := opentracing.Tags{
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     C.GoString(cbInfo.functionName),
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
	}
	if cbInfo.symbolName != nil {
		tags["symbol_name"] = C.GoString(cbInfo.symbolName)
	}
	span, ctx := c.tracer.StartSpanFromContext(c.ctx, "cupti_operation", tags)
	c.ctx = setSpanContextCorrelationId(ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCudaConfigureCallExit(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	span.Finish()
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
	tags := opentracing.Tags{
		"context_uid":       uint32(cbInfo.contextUid),
		"correlation_id":    correlationId,
		"function_name":     C.GoString(cbInfo.functionName),
		"cupti_domain":      domain.String(),
		"cupti_callback_id": cbid.String(),
		"stream":            uintptr(unsafe.Pointer(params.hStream)),
	}
	if cbInfo.symbolName != nil {
		tags["kernel"] = C.GoString(cbInfo.symbolName)
	}
	span, ctx := c.tracer.StartSpanFromContext(c.ctx, "cupti_operation", tags)
	c.ctx = setSpanContextCorrelationId(ctx, correlationId, span)

	return nil
}

func (c *CUPTI) onCULaunchKernelExit(domain types.CUpti_CallbackDomain, cbid types.CUpti_driver_api_trace_cbid, cbInfo *C.CUpti_CallbackData) error {
	correlationId := uint(cbInfo.correlationId)
	span, err := spanFromContextCorrelationId(c.ctx, correlationId)
	if err != nil {
		return err
	}
	span.Finish()
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

func (c *CUPTI) onCudaSetupArgument(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	return nil

}

func (c *CUPTI) onCudaLaunch(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
	return nil

}

func (c *CUPTI) onCudaMemCopy(domain types.CUpti_CallbackDomain, cbid types.CUPTI_RUNTIME_TRACE_CBID, cbInfo *C.CUpti_CallbackData) error {
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
			panic("handle device memcpy case....")
			return
		default:
			entry := log.WithField("cbid", cbid.String())
			if cbInfo.functionName != nil {
				entry = entry.WithField("function_name", C.GoString(cbInfo.functionName))
			}
			entry.Debug("skipping runtime call")
			return
		}
	case types.CUPTI_CB_DOMAIN_RUNTIME_API:
		cbid := types.CUPTI_RUNTIME_TRACE_CBID(cbid0)
		switch cbid {
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, types.CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
			handle.onCudaMemCopy(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
			handle.onCudaLaunch(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
			handle.onCudaConfigureCall(domain, cbid, cbInfo)
			return
		case types.CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
			handle.onCudaSetupArgument(domain, cbid, cbInfo)
			return
		default:
			entry := log.WithField("cbid", cbid.String())
			if cbInfo.functionName != nil {
				entry = entry.WithField("function_name", C.GoString(cbInfo.functionName))
			}
			entry.Debug("skipping runtime call")
			return
		}
	}
}
