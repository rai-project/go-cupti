package main

import "C"
import (
	"github.com/k0kubun/pp"
	"github.com/rai-project/go-cupti"
)

func main() {
	// config.Init()
	pp.Println(cupti.Version())
	// cupti.GetActivityObjectKindId(types.CUPTI_ACTIVITY_OBJECT_PROCESS, nil)
}
