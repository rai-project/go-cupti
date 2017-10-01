package cupti

// #include <cupti.h>
import "C"

var (
	DefaultCallbacks = []string{}
)

const (
	BUFFER_SIZE = 32 * 16384
	ALIGN_SIZE  = 8
)

// func (c *CUPTI) AddCallback() error {

// }

//export bufferRequested
func bufferRequested(buffer **C.uint8_t, size *C.size_t,
	maxNumRecords *C.size_t) {
	*size = BUFFER_SIZE + ALIGN_SIZE
	*buffer = C.calloc(1, *size)
	if *buffer == nil {
		panic("ran out of memory while performing bufferRequested")
	}
	*maxNumRecords = 0
}

//export bufferCompleted
func bufferCompleted(ctx C.CUcontext, streamId C.uint32_t, buffer *C.uint8_t,
	size C.size_t, validSize C.size_t) {

}
