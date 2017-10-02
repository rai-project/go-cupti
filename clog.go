// +build linux,cgo

package cupti

import "C"

//export log_error
func log_error(msg *C.char) {
	log.Error(msg)
}

//export log_info
func log_info(msg *C.char) {
	log.Info(msg)
}

//export log_debug
func log_debug(msg *C.char) {
	log.Debug(msg)
}
