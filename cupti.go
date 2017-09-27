package cupti

import (
	"time"

	tr "github.com/rai-project/tracer"
	context "golang.org/x/net/context"
)

type CUPTI struct {
	ctx             context.Context
	tracer          tr.Tracer
	deviceResetTime time.Time
}
