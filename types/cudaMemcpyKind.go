//go:generate enumer -type=CUDAMemcpyKind -json
package types

type CUDAMemcpyKind int

const (
	CUDAMemcpyHostToHost CUDAMemcpyKind = 0
	CUDAMemcpyHostToDevice
	CUDAMemcpyDeviceToHost
	CUDAMemcpyDeviceToDevice
)
