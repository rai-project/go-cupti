// +build !linux !cgo

package cuptigrpc

import (
	cupti "github.com/rai-project/go-cupti"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(optFuncs ...cupti.Option) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (resp interface{}, err error) {
		return handler(ctx, req)
	}
}
