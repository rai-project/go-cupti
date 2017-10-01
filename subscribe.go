package cupti

/*
#include <cupti.h>

extern void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo);
*/
import "C"
import "unsafe"

type callbackFunction func(userData unsafe.Pointer, domain0 C.CUpti_CallbackDomain, cbid0 C.CUpti_CallbackId, cbInfo *C.CUpti_CallbackData)

func cuptiGetTimestamp() (uint64, error) {
	var val C.uint64_t
	err := checkCUPTIError(C.cuptiGetTimestamp(&val))
	if err != nil {
		return 0, err
	}
	return uint64(val), nil
}

func cuptiEnableDomain() {

}

func (c *CUPTI) cuptiSubscribe() error {
	return checkCUPTIError(C.cuptiSubscribe(&c.subscriber, (C.CUpti_CallbackFunc)(unsafe.Pointer(C.callback)), unsafe.Pointer(c)))
}

func cuptiUnsubscribe() {

}
