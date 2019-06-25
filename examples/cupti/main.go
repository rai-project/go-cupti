package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -lcuda -lcudart -L/usr/local/cuda/lib64
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
import "C"

import (
	"context"
	"os"
	"path/filepath"
	"time"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/rai-project/config"
	"github.com/rai-project/go-cupti"
	"github.com/rai-project/logger"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/jaeger"
	_ "github.com/rai-project/tracer/noop"
	_ "github.com/rai-project/tracer/zipkin"
	"github.com/rainycape/dl"
)

var (
	log = logger.New().WithField("bin", "cupti")
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("bin", "cupti")
	})
}

var configOptions = []config.Option{
	config.AppName("carml"),
	config.ColorMode(true),
	config.DebugMode(true),
	config.VerboseMode(true),
}

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
	config.Init(configOptions...)
	version, err := cupti.Version()
	if err != nil {
		os.Exit(-1)
	}
	log.WithField("version", version).Debug("running cupti")

	ctx := context.Background()

	defer tracer.Close()

	func() {
		span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "vector_add")
		defer span.Finish()

		cupti, err := cupti.New(
			cupti.Context(ctx),
			cupti.Activities(nil),
			cupti.Callbacks([]string{
				"CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020",
				"CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000",
			}),
			cupti.Metrics(
				[]string{
					"flop_count_sp",
					"dram_read_bytes",
					"dram_write_bytes",
				},
			),
			cupti.Events(
				[]string{
					// "inst_executed",
				},
			))
		if err != nil {
			log.WithError(err).Error("failed to create new cupti context")
			os.Exit(-1)
		}
		defer cupti.Close()

		for ii := 0; ii < 1; ii++ {
			vectorAdd()
			C.cudaDeviceSynchronize()
		}
	}()
	time.Sleep(1 * time.Second)
}
