package cupti

import (
	context "context"
)

type Options struct {
	ctx            context.Context
	samplingPeriod int
	activities     []string
	domains        []string
	callbacks      []string
	events         []string
}

type Option func(o *Options)

func Context(ctx context.Context) Option {
	return func(o *Options) {
		o.ctx = ctx
	}
}

func Activities(activities []string) Option {
	if activities == nil {
		activities = []string{}
	}
	return func(o *Options) {
		o.activities = activities
	}
}

func Domains(domains []string) Option {
	if domains == nil {
		domains = []string{}
	}
	return func(o *Options) {
		o.domains = domains
	}
}

func Callbacks(callbacks []string) Option {
	if callbacks == nil {
		callbacks = []string{}
	}
	return func(o *Options) {
		o.callbacks = callbacks
	}
}

func Events(events []string) Option {
	if events == nil {
		events = []string{}
	}
	return func(o *Options) {
		o.events = events
	}
}

func SamplingPeriod(s int) Option {
	return func(o *Options) {
		o.samplingPeriod = s
	}
}

func NewOptions(opts ...Option) *Options {
	Config.Wait()

	options := &Options{
		ctx:            context.Background(),
		samplingPeriod: Config.SamplingPeriod,
		activities:     Config.Activities,
		domains:        Config.Domains,
		callbacks:      Config.Callbacks,
		events:         Config.Events,
	}

	for _, o := range opts {
		o(options)
	}

	return options
}
