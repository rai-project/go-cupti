package cupti

import (
	"encoding/json"
	"sync"

	"github.com/iancoleman/strcase"
	"github.com/pkg/errors"
)

// easyjson:json
type Event struct {
	Event            int    `json:"event,omitempty"`
	ID               int    `json:"id,omitempty"`
	DomainID         int    `json:"domain_id,omitempty"`
	Name             string `json:"name,omitempty"`
	ShortDescription string `json:"short_description,omitempty"`
	LongDescription  string `json:"long_description,omitempty"`
	Category         string `json:"category,omitempty"`
}

// easyjson:json
type events []Event

var AvailableEvents events = nil

func loadAvailableEvents() {
	bts, err := ReadFile("/event_mapping.json")
	if err != nil {
		panic(err)
	}
	var res events
	err = json.Unmarshal(bts, &res)
	if err != nil {
		panic(err)
	}
	AvailableEvents = res
}

func initAvailableEvents() {
	var once sync.Once
	once.Do(loadAvailableEvents)
}

func GetAvailableEvents() events {
	initAvailableEvents()
	return AvailableEvents
}

func FindEventByName(s0 string) (Event, error) {
	s := strcase.ToSnake(s0)
	for _, event := range AvailableEvents {
		if event.Name == s {
			return event, nil
		}
	}
	return Event{}, errors.Errorf("cannot find event with name %s", s0)
}
