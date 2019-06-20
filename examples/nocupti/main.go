package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -lcuda -lcudart -L/usr/local/cuda/lib64
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
import "C"

import (
	"path/filepath"
	"sync"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/rainycape/dl"
)

func vectorAdd() {
	srcDir := sourcepath.MustAbsoluteDir()
	vectorAddLibraryPath := filepath.Join(srcDir, "..", "..", "examples", "vector_add", "vector_add.so")
	if !com.IsFile(vectorAddLibraryPath) {
		panic("file " + vectorAddLibraryPath + " not found. make sure to compile the shared library")
	}
	dlib, err := dl.Open(vectorAddLibraryPath, 0)
	if err != nil {
		panic(err)
	}
	var vectorAdd func()
	err = dlib.Sym("VectorAdd", &vectorAdd)
	if err != nil {
		panic(err)
	}
	vectorAdd()
}

func main() {
	func() {
		var wg sync.WaitGroup
		for ii := 0; ii < 1; ii++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				C.cudaProfilerStart()
				vectorAdd()
				C.cudaDeviceSynchronize()
				C.cudaProfilerStop()
			}()
		}
		wg.Wait()

	}()
}
