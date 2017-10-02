// +build linux,cgo

package cuptigrpc

import (
	"github.com/apex/log"
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

		cupti, err := cupti.New(cupti.Context(ctx), cupti.Tracer(tracer))
		if err == nil {
			defer cupti.Close()
		} else {
			log.WithError(err).Error("failed to create new cupti context")
		}

		return handler(ctx, req)
	}
}
