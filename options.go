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
	metrics        []string
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
		o.activities = strUnion(activities)
	}
}

func Domains(domains []string) Option {
	if domains == nil {
		domains = []string{}
	}
	return func(o *Options) {
		o.domains = strUnion(domains)
	}
}

func Callbacks(callbacks []string) Option {
	if callbacks == nil {
		callbacks = []string{}
	}
	return func(o *Options) {
		o.callbacks = strUnion(callbacks)
	}
}

func Events(events []string) Option {
	if events == nil {
		events = []string{}
	}
	return func(o *Options) {
		o.events = strUnion(events)
		if len(o.events) == 0 {
			return
		}
		// maybe this is too automated, but we depend on the following
		// callbacks to run for events to work
		extraCallbacks := []string{
			"CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
			"CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING",
		}
		o.callbacks = strUnion(append(o.callbacks, extraCallbacks...))
	}
}

func Metrics(metrics []string) Option {
	if metrics == nil {
		metrics = []string{}
	}
	return func(o *Options) {
		o.metrics = strUnion(metrics)
		if len(o.metrics) == 0 {
			return
		}
		// maybe this is too automated, but we depend on the following
		// callbacks to run for events to work
		extraCallbacks := []string{
			"CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
			"CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING",
		}
		o.callbacks = strUnion(append(o.callbacks, extraCallbacks...))
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

func strUnion(inputs []string) []string {
	res := make([]string, 0, len(inputs))
	for _, input := range inputs {
		found := false
		for _, r := range res {
			if r == input {
				found = true
				break
			}
		}
		if !found {
			res = append(res, input)
		}
	}
	return res
}
