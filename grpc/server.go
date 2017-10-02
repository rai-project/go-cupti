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
		err = cupti.Subscribe()
		if err != nil {
			return handler(ctx, req)
		}
		defer cupti.Unsubscribe()
		defer cupti.Wait()
		tracer := cupti.Tracer()
		span, ctx := tracer.StartSpanFromContext(ctx, "cupti")
		defer span.Finish()
		cupti.SetContext(ctx)
		return handler(ctx, req)
	}
}
