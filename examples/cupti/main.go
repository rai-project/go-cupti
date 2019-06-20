package main

import "C"
import (
	"context"
	"os"
	"path/filepath"
	"sync"
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
				// "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel",
				"CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020",
				"CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000",
				// "CUPTI_CBID_NVTX_nvtxRangeStartA",
				// "CUPTI_CBID_NVTX_nvtxRangeStartEx",
				// "CUPTI_CBID_NVTX_nvtxRangeEnd",
				// "CUPTI_CBID_NVTX_nvtxRangePushA",
				// "CUPTI_CBID_NVTX_nvtxRangePushEx",
				// "CUPTI_CBID_NVTX_nvtxRangePop",
			}),
			cupti.Metrics(
				[]string{
					// "flop_count_sp",
					"dram_read_transactions",
					// "dram_write_transactions",
				},
			),
			cupti.Events(
				[]string{
					// "inst_executed",
					// "inst_executed_fma_pipe_s0",
					// "inst_executed_fma_pipe_s1",
					// "inst_executed_fma_pipe_s2",
					// "inst_executed_fma_pipe_s3",
					// "fb_subp0_read_sectors",
					// "fb_subp1_read_sectors",
					// "fb_subp0_write_sectors",
					// "fb_subp1_write_sectors",
				},
			))
		if err != nil {
			log.WithError(err).Error("failed to create new cupti context")
			os.Exit(-1)
		}
		defer cupti.Close()

		var wg sync.WaitGroup
		for ii := 0; ii < 1; ii++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				vectorAdd()
			}()
		}
		wg.Wait()

	}()
	time.Sleep(5 * time.Second)
}
