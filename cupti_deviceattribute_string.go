// Code generated by "enumer -type=CUpti_DeviceAttribute -json"; DO NOT EDIT

package cupti

import (
	"encoding/json"
	"fmt"
)

const (
	_CUpti_DeviceAttribute_name_0 = "CUPTI_DEVICE_ATTR_MAX_EVENT_IDCUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_IDCUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTHCUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLECUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISIONCUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERSCUPTI_DEVICE_ATTR_PCIE_LINK_RATECUPTI_DEVICE_ATTR_PCIE_LINK_WIDTHCUPTI_DEVICE_ATTR_PCIE_GENCUPTI_DEVICE_ATTR_DEVICE_CLASSCUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLECUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLECUPTI_DEVICE_ATTR_MAX_L2_UNITSCUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHAREDCUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUALCUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLECUPTI_DEVICE_ATTR_NVLINK_PRESENTCUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW"
	_CUpti_DeviceAttribute_name_1 = "CUPTI_DEVICE_ATTR_FORCE_INT"
)

var (
	_CUpti_DeviceAttribute_index_0 = [...]uint16{0, 30, 67, 108, 147, 204, 239, 271, 304, 330, 360, 395, 430, 460, 522, 580, 641, 676, 708, 743}
	_CUpti_DeviceAttribute_index_1 = [...]uint8{0, 27}
)

func (i CUpti_DeviceAttribute) String() string {
	switch {
	case 1 <= i && i <= 19:
		i -= 1
		return _CUpti_DeviceAttribute_name_0[_CUpti_DeviceAttribute_index_0[i]:_CUpti_DeviceAttribute_index_0[i+1]]
	case i == 2147483647:
		return _CUpti_DeviceAttribute_name_1
	default:
		return fmt.Sprintf("CUpti_DeviceAttribute(%d)", i)
	}
}

var _CUpti_DeviceAttributeNameToValue_map = map[string]CUpti_DeviceAttribute{
	_CUpti_DeviceAttribute_name_0[0:30]:    1,
	_CUpti_DeviceAttribute_name_0[30:67]:   2,
	_CUpti_DeviceAttribute_name_0[67:108]:  3,
	_CUpti_DeviceAttribute_name_0[108:147]: 4,
	_CUpti_DeviceAttribute_name_0[147:204]: 5,
	_CUpti_DeviceAttribute_name_0[204:239]: 6,
	_CUpti_DeviceAttribute_name_0[239:271]: 7,
	_CUpti_DeviceAttribute_name_0[271:304]: 8,
	_CUpti_DeviceAttribute_name_0[304:330]: 9,
	_CUpti_DeviceAttribute_name_0[330:360]: 10,
	_CUpti_DeviceAttribute_name_0[360:395]: 11,
	_CUpti_DeviceAttribute_name_0[395:430]: 12,
	_CUpti_DeviceAttribute_name_0[430:460]: 13,
	_CUpti_DeviceAttribute_name_0[460:522]: 14,
	_CUpti_DeviceAttribute_name_0[522:580]: 15,
	_CUpti_DeviceAttribute_name_0[580:641]: 16,
	_CUpti_DeviceAttribute_name_0[641:676]: 17,
	_CUpti_DeviceAttribute_name_0[676:708]: 18,
	_CUpti_DeviceAttribute_name_0[708:743]: 19,
	_CUpti_DeviceAttribute_name_1[0:27]:    2147483647,
}

func CUpti_DeviceAttributeString(s string) (CUpti_DeviceAttribute, error) {
	if val, ok := _CUpti_DeviceAttributeNameToValue_map[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_DeviceAttribute values", s)
}

func (i CUpti_DeviceAttribute) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

func (i *CUpti_DeviceAttribute) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_DeviceAttribute should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_DeviceAttributeString(s)
	return err
}
