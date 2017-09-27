package cupti

/*
#cgo CFLAGS: -I . -I /usr/local/cuda/include -I /usr/local/cuda/extras/CUPTI/include -DFMT_HEADER_ONLY
#cgo LDFLAGS: -L . -L /usr/local/cuda/lib -L /usr/local/cuda/extras/CUPTI/lib -lcupti -lcudart
*/
import "C"
