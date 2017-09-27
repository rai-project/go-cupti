// //go:generate enumer -type=CUptiResult -json

package types

type CUptiResult int

const (
	/**
	 * No error.
	 */
	CUPTI_SUCCESS CUptiResult = 0
	/**
	 * One or more of the parameters is invalid.
	 */
	CUPTI_ERROR_INVALID_PARAMETER CUptiResult = 1
	/**
	 * The device does not correspond to a valid CUDA device.
	 */
	CUPTI_ERROR_INVALID_DEVICE CUptiResult = 2
	/**
	 * The context is NULL or not valid.
	 */
	CUPTI_ERROR_INVALID_CONTEXT CUptiResult = 3
	/**
	 * The event domain id is invalid.
	 */
	CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID CUptiResult = 4
	/**
	 * The event id is invalid.
	 */
	CUPTI_ERROR_INVALID_EVENT_ID CUptiResult = 5
	/**
	 * The event name is invalid.
	 */
	CUPTI_ERROR_INVALID_EVENT_NAME CUptiResult = 6
	/**
	 * The current operation cannot be performed due to dependency on
	 * other factors.
	 */
	CUPTI_ERROR_INVALID_OPERATION CUptiResult = 7
	/**
	 * Unable to allocate enough memory to perform the requested
	 * operation.
	 */
	CUPTI_ERROR_OUT_OF_MEMORY CUptiResult = 8
	/**
	 * An error occurred on the performance monitoring hardware.
	 */
	CUPTI_ERROR_HARDWARE CUptiResult = 9
	/**
	 * The output buffer size is not sufficient to return all
	 * requested data.
	 */
	CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT CUptiResult = 10
	/**
	 * API is not implemented.
	 */
	CUPTI_ERROR_API_NOT_IMPLEMENTED CUptiResult = 11
	/**
	 * The maximum limit is reached.
	 */
	CUPTI_ERROR_MAX_LIMIT_REACHED CUptiResult = 12
	/**
	 * The object is not yet ready to perform the requested operation.
	 */
	CUPTI_ERROR_NOT_READY CUptiResult = 13
	/**
	 * The current operation is not compatible with the current state
	 * of the object
	 */
	CUPTI_ERROR_NOT_COMPATIBLE CUptiResult = 14
	/**
	 * CUPTI is unable to initialize its connection to the CUDA
	 * driver.
	 */
	CUPTI_ERROR_NOT_INITIALIZED CUptiResult = 15
	/**
	 * The metric id is invalid.
	 */
	CUPTI_ERROR_INVALID_METRIC_ID CUptiResult = 16
	/**
	 * The metric name is invalid.
	 */
	CUPTI_ERROR_INVALID_METRIC_NAME CUptiResult = 17
	/**
	 * The queue is empty.
	 */
	CUPTI_ERROR_QUEUE_EMPTY CUptiResult = 18
	/**
	 * Invalid handle (internal?).
	 */
	CUPTI_ERROR_INVALID_HANDLE CUptiResult = 19
	/**
	 * Invalid stream.
	 */
	CUPTI_ERROR_INVALID_STREAM CUptiResult = 20
	/**
	 * Invalid kind.
	 */
	CUPTI_ERROR_INVALID_KIND CUptiResult = 21
	/**
	 * Invalid event value.
	 */
	CUPTI_ERROR_INVALID_EVENT_VALUE CUptiResult = 22
	/**
	 * CUPTI is disabled due to conflicts with other enabled profilers
	 */
	CUPTI_ERROR_DISABLED CUptiResult = 23
	/**
	 * Invalid module.
	 */
	CUPTI_ERROR_INVALID_MODULE CUptiResult = 24
	/**
	 * Invalid metric value.
	 */
	CUPTI_ERROR_INVALID_METRIC_VALUE CUptiResult = 25
	/**
	 * The performance monitoring hardware is in use by other client.
	 */
	CUPTI_ERROR_HARDWARE_BUSY CUptiResult = 26
	/**
	 * The attempted operation is not supported on the current
	 * system or device.
	 */
	CUPTI_ERROR_NOT_SUPPORTED CUptiResult = 27
	/**
	 * Unified memory profiling is not supported on the system.
	 * Potential reason could be unsupported OS or architecture.
	 */
	CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED CUptiResult = 28
	/**
	 * Unified memory profiling is not supported on the device
	 */
	CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE CUptiResult = 29
	/**
	 * Unified memory profiling is not supported on a multi-GPU
	 * configuration without P2P support between any pair of devices
	 */
	CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES CUptiResult = 30
	/**
	 * Unified memory profiling is not supported under the
	 * Multi-Process Service (MPS) environment. CUDA 7.5 removes this
	 * restriction.
	 */
	CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS CUptiResult = 31
	/**
	 * An unknown internal error has occurred.
	 */
	CUPTI_ERROR_UNKNOWN   CUptiResult = 999
	CUPTI_ERROR_FORCE_INT CUptiResult = 0x7fffffff
)
