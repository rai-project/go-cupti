package cupti

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	tr "github.com/rai-project/tracer"
)

var (
	tracer tr.Tracer
	log    = logger.New().WithField("pkg", "go-cupti")
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "go-cupti")
		tracer = tr.MustNew("pkg/cupti")
	})
}
