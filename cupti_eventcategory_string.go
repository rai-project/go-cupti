// Code generated by "enumer -type=CUpti_EventCategory -json"; DO NOT EDIT

package cupti

import (
	"encoding/json"
	"fmt"
)

const (
	_CUpti_EventCategory_name_0 = "CUPTI_EVENT_CATEGORY_INSTRUCTIONCUPTI_EVENT_CATEGORY_MEMORYCUPTI_EVENT_CATEGORY_CACHECUPTI_EVENT_CATEGORY_PROFILE_TRIGGER"
	_CUpti_EventCategory_name_1 = "CUPTI_EVENT_CATEGORY_FORCE_INT"
)

var (
	_CUpti_EventCategory_index_0 = [...]uint8{0, 32, 59, 85, 121}
	_CUpti_EventCategory_index_1 = [...]uint8{0, 30}
)

func (i CUpti_EventCategory) String() string {
	switch {
	case 0 <= i && i <= 3:
		return _CUpti_EventCategory_name_0[_CUpti_EventCategory_index_0[i]:_CUpti_EventCategory_index_0[i+1]]
	case i == 2147483647:
		return _CUpti_EventCategory_name_1
	default:
		return fmt.Sprintf("CUpti_EventCategory(%d)", i)
	}
}

var _CUpti_EventCategoryNameToValue_map = map[string]CUpti_EventCategory{
	_CUpti_EventCategory_name_0[0:32]:   0,
	_CUpti_EventCategory_name_0[32:59]:  1,
	_CUpti_EventCategory_name_0[59:85]:  2,
	_CUpti_EventCategory_name_0[85:121]: 3,
	_CUpti_EventCategory_name_1[0:30]:   2147483647,
}

func CUpti_EventCategoryString(s string) (CUpti_EventCategory, error) {
	if val, ok := _CUpti_EventCategoryNameToValue_map[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_EventCategory values", s)
}

func (i CUpti_EventCategory) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

func (i *CUpti_EventCategory) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_EventCategory should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_EventCategoryString(s)
	return err
}
