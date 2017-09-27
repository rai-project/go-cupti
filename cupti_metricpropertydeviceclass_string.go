// Code generated by "enumer -type=CUpti_MetricPropertyDeviceClass -json"; DO NOT EDIT

package cupti

import (
	"encoding/json"
	"fmt"
)

const _CUpti_MetricPropertyDeviceClass_name = "CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLACUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADROCUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCECUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA"

var _CUpti_MetricPropertyDeviceClass_index = [...]uint8{0, 40, 81, 123, 163}

func (i CUpti_MetricPropertyDeviceClass) String() string {
	if i < 0 || i >= CUpti_MetricPropertyDeviceClass(len(_CUpti_MetricPropertyDeviceClass_index)-1) {
		return fmt.Sprintf("CUpti_MetricPropertyDeviceClass(%d)", i)
	}
	return _CUpti_MetricPropertyDeviceClass_name[_CUpti_MetricPropertyDeviceClass_index[i]:_CUpti_MetricPropertyDeviceClass_index[i+1]]
}

var _CUpti_MetricPropertyDeviceClassNameToValue_map = map[string]CUpti_MetricPropertyDeviceClass{
	_CUpti_MetricPropertyDeviceClass_name[0:40]:    0,
	_CUpti_MetricPropertyDeviceClass_name[40:81]:   1,
	_CUpti_MetricPropertyDeviceClass_name[81:123]:  2,
	_CUpti_MetricPropertyDeviceClass_name[123:163]: 3,
}

func CUpti_MetricPropertyDeviceClassString(s string) (CUpti_MetricPropertyDeviceClass, error) {
	if val, ok := _CUpti_MetricPropertyDeviceClassNameToValue_map[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_MetricPropertyDeviceClass values", s)
}

func (i CUpti_MetricPropertyDeviceClass) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

func (i *CUpti_MetricPropertyDeviceClass) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_MetricPropertyDeviceClass should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_MetricPropertyDeviceClassString(s)
	return err
}
