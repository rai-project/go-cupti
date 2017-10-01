package cupti

// #include <cupti.h>
import "C"
import (
	"time"

	"github.com/pkg/errors"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	tr "github.com/rai-project/tracer"
	context "golang.org/x/net/context"

	_ "github.com/rai-project/tracer/jaeger"
	_ "github.com/rai-project/tracer/noop"
	_ "github.com/rai-project/tracer/zipkin"
)

type CUPTI struct {
	ctx             context.Context
	tracer          tr.Tracer
	deviceResetTime time.Time
	subscriber      C.CUpti_SubscriberHandle
}

func New(ctx context.Context) (*CUPTI, error) {
	nvidiasmi.Wait()
	if !nvidiasmi.HasGPU {
		return nil, errors.New("no gpu found while trying to initialize cupti")
	}
	c := &CUPTI{
		ctx:             ctx,
		tracer:          tr.MustNew("cupti"),
		deviceResetTime: time.Now(),
	}

	_, err := c.DeviceReset()
	if err != nil {
		return nil, err
	}

	return c, nil
}
