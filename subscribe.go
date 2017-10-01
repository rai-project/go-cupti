package cupti

// #include <cupti.h>
import "C"
import "unsafe"

func cuptiGetTimestamp() {

}

func cuptiEnableCallback() {

}

func cuptiEnableDomain() {

}

func cuptiSubscribe(c *CUPTI, callback func()) error {
	return checkCUPTIError(C.cuptiSubscribe(&c.subscriber, C.CUpti_CallbackFunc(unsafe.Pointer(&callback)), unsafe.Pointer(c)))
}

func cuptiUnsubscribe() {

}
