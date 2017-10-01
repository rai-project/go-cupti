package cupti

// #include <cupti.h>
import "C"

func (c *CUPTI) Close() error {
	if c.subscriber == nil {
		return nil
	}
	C.cuptiUnsubscribe(c.subscriber)
	return nil
}
