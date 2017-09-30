package cupti

import (
	"testing"

	"github.com/rai-project/go-cupti/types"
)

func Test1(t *testing.T) {
	getActivityObjectKindId(types.CUPTI_ACTIVITY_OBJECT_PROCESS, nil)
}
