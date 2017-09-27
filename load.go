package cupti

import (
	"github.com/rai-project/config"
	"github.com/rai-project/nvidia-smi"
	"github.com/rainycape/dl"
)

var (
	dlib *dl.DL
)

func load() {
	var err error
	dlib, err = dl.Open("cupti", 0)
	if err != nil {
		log.WithError(err).Error("unable to find cupti")
	}
}

func init() {
	config.AfterInit(func() {
		nvidiasmi.Wait()
		if !nvidiasmi.HasGPU {
			return
		}
		load()
	})
}
