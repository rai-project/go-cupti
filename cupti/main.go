package main

import "C"
import (
	"os"

	"github.com/k0kubun/pp"
	"github.com/rai-project/go-cupti"
)

func main() {
	// config.Init()
	version, err := cupti.Version()
	if err != nil {
		os.Exit(-1)
	}
	pp.Println(int(version))
	_ = cupti.GetActivityObjectKindId()
	// cupti.GetActivityObjectKindId(types.CUPTI_ACTIVITY_OBJECT_PROCESS, nil)
}
