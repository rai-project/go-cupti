// +build linux,cgo,!arm64

package cupti

/*
#cgo CFLAGS: -I${SRCDIR}/cbits -O3 -Wall -g
#cgo CXXFLAGS: -std=c++11
#cgo CFLAGS: -I .
#cgo CFLAGS: -I /usr/local/cuda/include -I /usr/local/cuda-9-2/include 
#cgo CFLAGS: -I /usr/local/cuda/extras/CUPTI/include -I /usr/local/cuda-9-2/extras/CUPTI/include
#cgo CFLAGS: -DFMT_HEADER_ONLY
#cgo LDFLAGS: -L . -lcuda -lcudart -lcupti -Wl,-rpath,$ORIGIN
#cgo darwin,amd64 LDFLAGS: -L /usr/local/cuda/lib -L /usr/local/cuda-9-2/lib  -L /usr/local/cuda/extras/CUPTI/lib  -L /usr/local/cuda-9-2/extras/CUPTI/lib
#cgo linux,amd64 linux,ppc64le LDFLAGS: -L /usr/local/cuda/lib64 -L /usr/local/cuda-9-2/lib64 -L /usr/local/cuda/extras/CUPTI/lib64 -L /usr/local/cuda-9-2/extras/CUPTI/lib64
*/
import "C"
