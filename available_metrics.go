package cupti

import (
	"encoding/json"
	"sync"

	"github.com/iancoleman/strcase"
	"github.com/pkg/errors"
)

// easyjson:json
type Metric struct {
	Metric           int    `json:"metric,omitempty"`
	ID               int    `json:"id,omitempty"`
	Name             string `json:"name,omitempty"`
	ShortDescription string `json:"short_description,omitempty"`
	LongDescription  string `json:"long_description,omitempty"`
}

// easyjson:json
type metrics []Metric

var AvailableMetrics metrics = nil

func loadAvailableMetrics() {
	bts, err := ReadFile("/metric_mapping.json")
	if err != nil {
		panic(err)
	}
	var res metrics
	err = json.Unmarshal(bts, &res)
	if err != nil {
		panic(err)
	}
	AvailableMetrics = res
}

func initAvailableMetrics() {
	var once sync.Once
	once.Do(loadAvailableMetrics)
}

func GetAvailableMetrics() metrics {
	initAvailableMetrics()
	return AvailableMetrics
}

func FindMetricByName(s0 string) (Metric, error) {
	s := strcase.ToSnake(s0)
	for _, metric := range AvailableMetrics {
		if metric.Name == s {
			return metric, nil
		}
	}
	return Metric{}, errors.Errorf("cannot find metric with name %s", s0)
}
