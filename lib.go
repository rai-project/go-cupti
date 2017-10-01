package cupti

/*
#cgo CFLAGS: -I${SRCDIR}/cbits -O3 -Wall -g
#cgo CXXFLAGS: -std=c++11
#cgo CFLAGS: -I . -I /usr/local/cuda/include -I /usr/local/cuda/extras/CUPTI/include -DFMT_HEADER_ONLY
#cgo LDFLAGS: -L . -lcuda -lcudart -lcupti -Wl,-rpath -Wl,$ORIGIN
#cgo darwin,amd64 LDFLAGS: -L /usr/local/cuda/lib -L /usr/local/cuda/extras/CUPTI/lib
#cgo linux,amd64 LDFLAGS: -L /usr/local/cuda/lib64 -L /usr/local/cuda/extras/CUPTI/lib64
#cgo linux,ppc64le LDFLAGS: -L /usr/local/cuda/lib -L /usr/local/cuda/extras/CUPTI/lib
*/
import "C"

// DISABLED #cgo darwin,amd64 LDFLAGS: -Wl,-rpath,/usr/local/cuda/lib,/usr/local/cuda/extras/CUPTI/lib
