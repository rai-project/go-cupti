// +build !linux !cgo arm64

package cupti

import (
	context "golang.org/x/net/context"
)

type noopCloser struct{}

func (noopCloser) Close() error {
	return nil
}

type CUPTI noopCloser

func New(opts ...Option) (*CUPTI, error) {
	return nil, nil
}

func (c *CUPTI) Wait() error {
	return nil
}

func (c *CUPTI) Close() error {
	return nil
}

func (c *CUPTI) SetContext(ctx context.Context) {
}

func (c *CUPTI) Subscribe() error {
	return nil
}

func (c *CUPTI) Unsubscribe() error {
	return nil
}
