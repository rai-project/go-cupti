// +build linux,cgo,!arm64

package cuptigrpc

import (
	"context"
	opentracing "github.com/opentracing/opentracing-go"
	cupti "github.com/rai-project/go-cupti"
	"google.golang.org/grpc"
)

func ServerUnaryInterceptor(opts ...cupti.Option) grpc.UnaryServerInterceptor {
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
		parent := opentracing.SpanFromContext(ctx)
		if parent == nil {
			return handler(ctx, req)
		}
		tracer := parent.Tracer()
		if tracer == nil {
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
		defer cuptiHandle.Wait()
		defer cuptiHandle.Unsubscribe()
		ctx = context.WithValue(ctx, Handle{}, cuptiHandle)
		return handler(ctx, req)
	}
}
