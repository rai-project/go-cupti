package cupti

import (
	"encoding/json"
	"sync"

	"github.com/iancoleman/strcase"
	"github.com/pkg/errors"
)

type Event struct {
	Event            int    `json:"event"`
	ID               int    `json:"id"`
	DomainID         int    `json:"domain_id"`
	Name             string `json:"name"`
	ShortDescription string `json:"short_description"`
	LongDescription  string `json:"long_description"`
	Category         string `json:"category,omitempty"`
}

type events []Event

var Events events = nil

func loadEvents() {
	bts, err := ReadFile("/event_mapping.json")
	if err != nil {
		panic(err)
	}
	var res events
	err = json.Unmarshal(&res, bts)
	if err != nil {
		panic(err)
	}
	Events = &res
}

func initEvents() {
	var once sync.Once
	once.Do(loadEvents)
}

func GetEvents() Events {
	initEvents()
	return Events
}

func FindEventByName(s0 string) (Event, error) {
	s := strcase.ToSnake(s0)
	for _, event := range Events {
		if event.Name == s {
			return event
		}
	}
	return Event{}, errors.Errorf("cannot find event with name %s", s0)
}
