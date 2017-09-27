package cupti

import (
	"os/exec"
	"path/filepath"

	"github.com/Unknwon/com"

	"github.com/rai-project/config"
	"github.com/rai-project/nvidia-smi"
	"github.com/rainycape/dl"
)

var (
	dlib                    *dl.DL
	DefaultCUPTILibraryPath string
	DefaultCUDALibraryPath  string
)

func load() {
	var err error
	lib := filepath.Join(DefaultCUPTILibraryPath, "cupti"+dl.LibExt)
	if !com.IsFile(lib) {
		log.WithField("lib", lib).Error("unable to find cupti library")
		return
	}
	dlib, err = dl.Open(lib, 0)
	if err != nil {
		log.WithError(err).Error("unable to find cupti")
	}
}

func init() {
	config.AfterInit(func() {
		nvccPath, err := exec.LookPath("nvcc")
		if err != nil {
			nvccPath = "/usr/local/cuda/bin/nvcc"
		}
		cudaBaseDir, _ := filepath.Abs(filepath.Join(nvccPath, "..", ".."))
		if DefaultCUDALibraryPath == "" {
			DefaultCUDALibraryPath = filepath.Join(cudaBaseDir, "lib")
		}
		if DefaultCUPTILibraryPath == "" {
			DefaultCUPTILibraryPath = filepath.Join(cudaBaseDir, "extras", "CUPTI", "lib")
		}

		nvidiasmi.Wait()
		if !nvidiasmi.HasGPU {
			return
		}
		load()
	})
}
