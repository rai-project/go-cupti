package cupti

type noopCloser struct{}

func (noopCloser) Close() error {
	return nil
}

type CUPTI noopCloser

func New(opts ...Option) (*CUPTI, error) {
	return nil, nil
}

func (c *CUPTI) Close() error {
	return nil
}
