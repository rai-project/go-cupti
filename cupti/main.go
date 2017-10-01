package main

import "C"
import (
	"os"

	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/go-cupti"
)

var configOptions = []config.Option{
	config.AppName("carml"),
	config.ColorMode(true),
	config.DebugMode(true),
	config.VerboseMode(true),
}

func main() {
	config.Init(configOptions...)
	version, err := cupti.Version()
	if err != nil {
		os.Exit(-1)
	}
	pp.Println(version)
	// cupti.GetActivityObjectKindId(types.CUPTI_ACTIVITY_OBJECT_PROCESS, nil)
}
