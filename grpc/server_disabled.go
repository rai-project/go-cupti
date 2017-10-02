// +build !linux !cgo

package cuptigrpc

import (
	cupti "github.com/rai-project/go-cupti"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(_ ...cupti.Option) grpc.UnaryServerInterceptor {
	return noopUnaryServer
}
