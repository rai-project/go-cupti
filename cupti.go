package cupti

// #include <cupti.h>
import "C"
import (
	"sync"
	"time"

	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"

	_ "github.com/rai-project/tracer/jaeger"
	_ "github.com/rai-project/tracer/noop"
	_ "github.com/rai-project/tracer/zipkin"
)

type CUPTI struct {
	*Options
	subscriber      C.CUpti_SubscriberHandle
	deviceResetTime time.Time
	startTimeStamp  uint64
	beginTime       time.Time
}

func New(opts ...Option) (*CUPTI, error) {
	nvidiasmi.Wait()
	if !nvidiasmi.HasGPU {
		return nil, errors.New("no gpu found while trying to initialize cupti")
	}

	options := NewOptions(opts...)
	c := &CUPTI{
		Options: options,
	}
	if err := c.init(); err != nil {
		return nil, err
	}

	if err := c.startActivies(); err != nil {
		return nil, err
	}

	if err := c.registerCallbacks(); err != nil {
		return nil, err
	}

	startTimeStamp, err := cuptiGetTimestamp()
	if err != nil {
		return nil, err
	}
	c.startTimeStamp = startTimeStamp
	c.beginTime = time.Now()
	return c, nil
}

var cuInitOnce sync.Once

func (c *CUPTI) init() error {
	cuInitOnce.Do(func() {
		if err := checkCUResult(C.cuInit(0)); err != nil {
			log.WithError(err).Error("failed to perform cuInit")
			return
		}
		for id, gpu := range nvidiasmi.Info.GPUS {
			var cuCtx C.CUcontext

			if err := checkCUResult(C.cuCtxCreate(&cuCtx, 0, C.CUdevice(id))); err != nil {
				log.WithError(err).
					WithField("device_index", id).
					WithField("device_id", gpu.ID).
					Error("failed to create cuda context")
				return
			}
			// var samplingConfig C.CUpti_ActivityPCSamplingConfig
			// samplingConfig.samplingPeriod = C.CUpti_ActivityPCSamplingPeriod(c.samplingPeriod)
			// if err := cuptiActivityConfigurePCSampling(cuCtx, samplingConfig); err != nil {
			// 	log.WithError(err).WithField("device_id", gpu.ID).Error("failed to set cupti sampling period")
			// 	return
			// }
		}
	})

	if _, err := c.DeviceReset(); err != nil {
		return err
	}

	if err := c.cuptiSubscribe(); err != nil {
		return err
	}

	return nil
}

func (c *CUPTI) Close() error {
	if c == nil {
		return nil
	}

	c.Wait()

	if err := cuptiActivityFlushAll(); err != nil {
		log.WithError(err).Error("failed to flush all activities")
	}
	if err := c.stopActivies(); err != nil {
		log.WithError(err).Error("failed to stop activities")
	}
	if c.subscriber != nil {
		C.cuptiUnsubscribe(c.subscriber)
	}

	return nil
}

func (c *CUPTI) startActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrap(err, "unable to start activities")
		}
		err = cuptiActivityEnable(activity)
		if err != nil {
			return errors.Wrap(err, "unable to enable activities")
		}
	}
	return nil
}

func (c *CUPTI) stopActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrap(err, "unable to stop activities")
		}
		err = cuptiActivityDisable(activity)
		if err != nil {
			return errors.Wrap(err, "unable to disable activities")
		}
	}
	return nil
}

func (c *CUPTI) registerCallbacks() error {
	for _, callback := range c.callbacks {
		err := c.addCallback(callback)
		if err != nil {
			return err
		}
	}
	return nil
}
