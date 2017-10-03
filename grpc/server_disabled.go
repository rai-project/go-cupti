// +build !linux !cgo arm64

package cuptigrpc

import (
	cupti "github.com/rai-project/go-cupti"
	"github.com/rai-project/tracer"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(_ tracer.Tracer, _ ...cupti.Option) grpc.UnaryServerInterceptor {
	return noopUnaryServer
}
