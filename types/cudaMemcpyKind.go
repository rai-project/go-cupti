//go:generate enumer -type=CUDAMemcpyKind -json
package types

type CUDAMemcpyKind int

const (
	CUDAMemcpyHostToHost     CUDAMemcpyKind = 0
	CUDAMemcpyHostToDevice   CUDAMemcpyKind = 1
	CUDAMemcpyDeviceToHost   CUDAMemcpyKind = 2
	CUDAMemcpyDeviceToDevice CUDAMemcpyKind = 3
)
