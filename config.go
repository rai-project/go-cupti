package cupti

import (
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/vipertags"
)

type cuptiConfig struct {
	SamplingPeriod int           `json:"sampling_period" config:"cupti.sampling_period" default:5`
	Activities     []string      `json:"activities" config:"cupti.activities"`
	Domains        []string      `json:"domains" config:"cupti.domains"`
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
