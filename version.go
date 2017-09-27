package cupti

import (
	"sync"

	"github.com/pkg/errors"
)

var cuptiGetVersion func(*uint32) uint

func Version() (uint32, error) {
	var once sync.Once
	loadCuptiGetVersion := func() {
		err := dlib.Sym("cuptiGetVersion", &cuptiGetVersion)
		if err != nil {
			log.WithError(err).Error("Failed to load cuptiGetVersion")
		}
	}

	once.Do(loadCuptiGetVersion)
	if cuptiGetVersion == nil {
		return 0, errors.New("failed to load cuptiGetVersion")
	}
	var version uint32
	err := cuptiGetVersion(&version)
	if err != 0 {
		return 0, errors.Errorf("got error code = %d while calling cuptiGetVersion", err)
	}
	return version, nil
}
