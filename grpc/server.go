// +build linux,cgo

package cuptigrpc

import (
	cupti "github.com/rai-project/go-cupti"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(opts ...cupti.Option) grpc.UnaryServerInterceptor {
	cupti, err := cupti.New(opts...)
	if err != nil {
		return noopUnaryServer
	}
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (resp interface{}, err error) {
		cupti.Subscribe()
		defer cupti.Unsubscribe()
		defer cupti.Wait()
		cupti.SetContext(ctx)
		return handler(ctx, req)
	}
}
