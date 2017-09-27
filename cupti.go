package cupti

import (
	tr "github.com/rai-project/tracer"
	context "golang.org/x/net/context"
)

type CUPTI struct {
	ctx    context.Context
	tracer tr.Tracer
}
