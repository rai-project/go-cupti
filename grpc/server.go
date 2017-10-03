// +build linux,cgo,!arm64

package cuptigrpc

import (
	cupti "github.com/rai-project/go-cupti"
	"github.com/rai-project/tracer"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(tr tracer.Tracer, opts ...cupti.Option) grpc.UnaryServerInterceptor {
	opts = append([]cupti.Option{cupti.Tracer(tr)}, opts...)
	cuptiHandle, err := cupti.New(opts...)
	if err != nil {
		//pp.Println("noop unary")
		return noopUnaryServer
	}
	cuptiHandle.Unsubscribe()
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (resp interface{}, err error) {
		if cuptiHandle == nil {
			//	pp.Println("no cupti handle")
			return handler(ctx, req)
		}
		tracer := cuptiHandle.Tracer()
		if tracer == nil {
			//pp.Println("no cupti trace")
			return handler(ctx, req)
		}
		span, ctx := tracer.StartSpanFromContext(ctx, "cupti")
		defer span.Finish()
		cuptiHandle.SetContext(ctx)
		err = cuptiHandle.Subscribe()
		if err != nil {
			//pp.Println("failed to subscribe")
			return handler(ctx, req)
		}
		defer cuptiHandle.Unsubscribe()
		defer cuptiHandle.Wait()
		ctx = context.WithValue(ctx, Handle{}, cuptiHandle)
		return handler(ctx, req)
	}
}
