// +build linux,cgo,!arm64,!nogpu

package cupti

// #include <cupti.h>
import "C"
import (
	"context"
	"sync"
	"time"
	"unsafe"

	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/jaeger"
	_ "github.com/rai-project/tracer/noop"
	_ "github.com/rai-project/tracer/zipkin"
)

type CUPTI struct {
	*Options
	sync.RWMutex
	cuCtxs          []C.CUcontext
	subscriber      C.CUpti_SubscriberHandle
	deviceResetTime time.Time
	startTimeStamp  uint64
	beginTime       time.Time
  eventData       []*eventData
  spans sync.Map
}

type eventData struct {
	cuCtx      C.CUcontext
	cuCtxID    uint32
	deviceId   uint32
	eventGroup C.CUpti_EventGroup
	eventIds   map[string]C.CUpti_EventID
}

var (
	currentCUPTI *CUPTI
)

func New(opts ...Option) (*CUPTI, error) {
	nvidiasmi.Wait()
	if !nvidiasmi.HasGPU {
		return nil, errors.New("no gpu found while trying to initialize cupti")
	}

	runInit()

	options := NewOptions(opts...)
	c := &CUPTI{
		Options:   options,
		eventData: []*eventData{},
	}

	span, _ := tracer.StartSpanFromContext(options.ctx, tracer.FULL_TRACE, "cupti_new")
	defer span.Finish()

	currentCUPTI = c

	if err := c.Subscribe(); err != nil {
		return nil, c.Close()
	}

	startTimeStamp, err := cuptiGetTimestamp()
	if err != nil {
		return nil, err
	}
	c.startTimeStamp = startTimeStamp
	c.beginTime = time.Now()

	return c, nil
}

func (c *CUPTI) SetContext(ctx context.Context) {
	c.Lock()
	defer c.Unlock()
	c.ctx = ctx
}

func runInit() {
	initAvailableEvents()

	if err := checkCUResult(C.cuInit(0)); err != nil {
		log.WithError(err).Error("failed to perform cuInit")
	}
}

func (c *CUPTI) Subscribe() error {
	if err := c.startActivies(); err != nil {
		return err
	}

	if err := c.cuptiSubscribe(); err != nil {
		return err
	}

	if err := c.enableCallbacks(); err != nil {
		return err
	}

	if c.samplingPeriod != 0 {
		for ii, cuCtx := range c.cuCtxs {
			var samplingConfig C.CUpti_ActivityPCSamplingConfig
			samplingConfig.samplingPeriod = C.CUpti_ActivityPCSamplingPeriod(c.samplingPeriod)
			if err := cuptiActivityConfigurePCSampling(cuCtx, samplingConfig); err != nil {
				log.WithError(err).WithField("device_id", ii).Error("failed to set cupti sampling period")
				return err
			}
		}
	}

	return nil
}

func (c *CUPTI) Unsubscribe() error {
	c.Wait()
	if err := c.stopActivies(); err != nil {
		return err
	}
	if err := c.deleteEventGroups(); err != nil {
		return err
	}
	if err := c.cuptiUnsubscribe(); err != nil {
		return err
	}
	return nil
}

func (c *CUPTI) Close() error {
	if c == nil {
		return nil
	}
	c.Unsubscribe()

	currentCUPTI = nil
	return nil
}

func (c *CUPTI) startActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrapf(err, "unable to map %v to activity kind", activityName)
		}
		err = cuptiActivityEnable(activity)
		if err != nil {
			log.WithError(err).
				WithField("activity", activityName).
				WithField("activity_enum", int(activity)).
				Error("unable to enable activity")
			panic(err)
			return errors.Wrapf(err, "unable to enable activitiy %v", activityName)
		}
	}

	err := cuptiActivityRegisterCallbacks()
	if err != nil {
		return errors.Wrap(err, "unable to register activity callbacks")
	}

	return nil
}

func (c *CUPTI) stopActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrapf(err, "unable to map %v to activity kind", activityName)
		}
		err = cuptiActivityDisable(activity)
		if err != nil {
			return errors.Wrapf(err, "unable to disable activity %v", activityName)
		}
	}
	return nil
}

func (c *CUPTI) enableCallbacks() error {
	for _, callback := range c.callbacks {
		err := c.enableCallback(callback)
		if err != nil {
			return err
		}
	}
	return nil
}

// not used
func (c *CUPTI) init() error {
	hasEvents := len(c.events) != 0
	c.cuCtxs = make([]C.CUcontext, len(nvidiasmi.Info.GPUS))
	if hasEvents {
		c.eventData = make([]*eventData, len(nvidiasmi.Info.GPUS))
	}
	for ii, gpu := range nvidiasmi.Info.GPUS {
		var cuCtx C.CUcontext

		if err := checkCUResult(C.cuCtxCreate(&cuCtx, 0, C.CUdevice(ii))); err != nil {
			log.WithError(err).
				WithField("device_index", ii).
				WithField("device_id", gpu.ID).
				Error("failed to create cuda context")
			return err
		}
		c.cuCtxs[ii] = cuCtx

		if hasEvents {
			ctxId := uint32(0)
			err := checkCUPTIError(C.cuptiGetContextId(cuCtx, (*C.uint32_t)(&ctxId)))
			if err != nil {
				return errors.Wrap(err, "unable to get device id when creating resource context")
			}

			eventData, err := c.createEventGroup(cuCtx, ctxId, uint32(ii))
			if err != nil {
				return errors.Errorf("cannot create event group for device %d", ii)
			}
			c.eventData[ii] = eventData
		}
	}

	if _, err := c.DeviceReset(); err != nil {
		return err
	}

	return nil
}

func (c *CUPTI) addEventGroup(cuCtx C.CUcontext, cuCtxID uint32, deviceId uint32) error {
	if len(c.events) == 0 {
		return nil
	}

	eventData, err := c.createEventGroup(cuCtx, cuCtxID, deviceId)
	if err != nil {
		err := errors.Wrapf(err, "cannot create event group for device %d", deviceId)
		log.WithError(err).WithField("device_id", deviceId).Error("cannot create event group")
		return err
	}

	c.eventData = append(c.eventData, eventData)
	return nil
}

func (c *CUPTI) findEventDataByCUCtxID(cuCtxID uint32) (*eventData, error) {
	for _, eventDataItem := range c.eventData {
		if eventDataItem == nil {
			continue
		}
		if eventDataItem.cuCtxID == cuCtxID {
			return eventDataItem, nil
		}
	}
	return nil, errors.Errorf("cannot find event group for cutxid = %d", cuCtxID)
}

func (c *CUPTI) removeEventGroup(cuCtx C.CUcontext, cuCtxID uint32, deviceId uint32) error {
	for ii, eventDataItem := range c.eventData {
		if eventDataItem == nil {
			continue
		}
		if eventDataItem.cuCtxID == cuCtxID && eventDataItem.deviceId == deviceId {
			c.deleteEventGroup(eventDataItem)
			c.eventData[ii] = nil
		}
	}
	return nil
}

func (c *CUPTI) createEventGroup(cuCtx C.CUcontext, cuCtxID uint32, deviceId uint32) (*eventData, error) {
	if len(c.events) == 0 {
		return nil, nil
	}

	var eventGroup C.CUpti_EventGroup

	err := checkCUPTIError(C.cuptiEventGroupCreate(cuCtx, &eventGroup, 0))
	if err != nil {
		return nil, errors.Wrapf(err, "cannot create event group")
	}

	eventIds := map[string]C.CUpti_EventID{}
	for _, userEventName := range c.events {
		eventName := userEventName
		eventInfo, err := FindEventByName(userEventName)
		if err == nil {
			eventName = eventInfo.Name
		}

		var eventId C.CUpti_EventID
		cEventName := C.CString(eventName)

		err = checkCUPTIError(C.cuptiEventGetIdFromName(C.CUdevice(deviceId), cEventName, &eventId))
		C.free(unsafe.Pointer(cEventName))
		if err != nil {
			return nil, errors.Wrapf(err, "cannot find event id %v", userEventName)
		}

		err = checkCUPTIError(C.cuptiEventGroupAddEvent(eventGroup, eventId))
		if err != nil {
			return nil, errors.Wrapf(err, "cannot add event %v to event group", userEventName)
		}

		eventIds[userEventName] = eventId

	}

	profileAll := C.uint(1)

	err = checkCUPTIError(C.cuptiEventGroupSetAttribute(
		eventGroup,
		C.CUpti_EventGroupAttribute(types.CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES),
		C.size_t(unsafe.Sizeof(profileAll)),
		unsafe.Pointer(&profileAll),
	))
	if err != nil {
		return nil, errors.Wrap(err, "cannot set event group attribute")
	}

	return &eventData{
		cuCtx:      cuCtx,
		cuCtxID:    cuCtxID,
		deviceId:   deviceId,
		eventGroup: eventGroup,
		eventIds:   eventIds,
	}, nil
}

func (c *CUPTI) deleteEventGroups() error {
	for ii, eventDataItem := range c.eventData {
		c.deleteEventGroup(eventDataItem)
		c.eventData[ii] = nil
	}
	return nil
}

func (c *CUPTI) deleteEventGroup(eventDataItem *eventData) error {
	if eventDataItem == nil {
		return nil
	}
	eventGroup := eventDataItem.eventGroup
	for eventName, eventId := range eventDataItem.eventIds {
		err := checkCUPTIError(C.cuptiEventGroupRemoveEvent(eventGroup, eventId))
		if err != nil {
			panic(err)
			log.WithError(err).WithField("event_name", eventName).Error("unable to remove event from event group")
		}
	}
	err := checkCUPTIError(C.cuptiEventGroupDestroy(eventGroup))
	if err != nil {
		log.WithError(err).Error("unable to remove event group")
	}
	return nil
}
