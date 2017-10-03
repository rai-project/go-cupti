// +build linux,amd64 

package cupti

import (
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/Unknwon/com"

	"github.com/rai-project/config"
	"github.com/rainycape/dl"
)

var (
	dlib                    *dl.DL
	DefaultCUPTILibraryPath string
	DefaultCUDALibraryPath  string
)

func load() {
	var err error
	libPrefix := ""
	if runtime.GOOS == "linux" {
		libPrefix = "lib"
	}
	lib := filepath.Join(DefaultCUPTILibraryPath, libPrefix+"cupti"+dl.LibExt)
	if !com.IsFile(lib) {
		log.WithField("lib", lib).Error("unable to find cupti library")
		return
	}
	log.WithField("lib", lib).Debug("loading cupti library")
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
			if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" {
				DefaultCUPTILibraryPath += "64"
			}
		}
		load()
	})
}
