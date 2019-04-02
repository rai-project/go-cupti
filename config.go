package cupti

import (
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/vipertags"
)

type cuptiConfig struct {
	SamplingPeriod int           `json:"sampling_period" config:"cupti.sampling_period"`
	Activities     []string      `json:"activities" config:"cupti.activities"`
	Domains        []string      `json:"domains" config:"cupti.domains"`
	Callbacks      []string      `json:"callbacks" config:"cupti.callbacks"`
	Events         []string      `json:"events" config:"cupti.events"`
	done           chan struct{} `json:"-" config:"-"`
}

var (
	Config = &cuptiConfig{
		done: make(chan struct{}),
	}
)

func (cuptiConfig) ConfigName() string {
	return "cupti"
}

func (a *cuptiConfig) SetDefaults() {
	vipertags.SetDefaults(a)
}

func (a *cuptiConfig) Read() {
	defer close(a.done)
	vipertags.Fill(a)
	if len(a.Activities) == 0 {
		a.Activities = DefaultActivities
	}
	if len(a.Domains) == 0 {
		a.Domains = DefaultDomains
	}
	if len(a.Callbacks) == 0 {
		a.Callbacks = DefaultCallbacks
	}
	if len(a.Events) == 0 {
		a.Events = DefaultEvents
	}
}

func (c cuptiConfig) Wait() {
	<-c.done
}

func (c cuptiConfig) String() string {
	return pp.Sprintln(c)
}

func (c cuptiConfig) Debug() {
	log.Debug("cupti Config = ", c)
}

func init() {
	config.Register(Config)
}
