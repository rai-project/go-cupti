// +build linux,cgo,!nogpu

package cupti

/*
#include <cupti.h>

extern void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const void *cbInfo);
*/
import "C"
import "unsafe"

type callbackFunction func(userData unsafe.Pointer, domain0 C.CUpti_CallbackDomain, cbid0 C.CUpti_CallbackId, cbInfo unsafe.Pointer)

func cuptiEnableDomain() {

}

func (c *CUPTI) cuptiSubscribe() error {
	var subscriber C.CUpti_SubscriberHandle

	err := checkCUPTIError(
		C.cuptiSubscribe(
			&subscriber,
			(C.CUpti_CallbackFunc)(unsafe.Pointer(C.callback)),
			unsafe.Pointer(c),
		),
	)
	if err != nil {
		return err
	}
	c.subscriber = subscriber
	return nil
}

func (c *CUPTI) cuptiUnsubscribe() error {
	if c.subscriber == nil {
		return nil
	}
	err := checkCUPTIError(C.cuptiUnsubscribe(c.subscriber))
	if err != nil {
		return err
	}
	c.subscriber = nil
	return nil
}
