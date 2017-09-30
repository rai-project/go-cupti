package cupti

/*
#cgo CFLAGS: -I . -I /usr/local/cuda/include -I /usr/local/cuda/extras/CUPTI/include -DFMT_HEADER_ONLY
#cgo LDFLAGS: -L . -lcupti -lcudart
#cgo amd64 LDFLAGS: -L /usr/local/cuda/lib64 -L /usr/local/cuda/extras/CUPTI/lib64
#cgo ppc64le LDFLAGS: -L /usr/local/cuda/lib -L /usr/local/cuda/extras/CUPTI/lib
*/
import "C"
