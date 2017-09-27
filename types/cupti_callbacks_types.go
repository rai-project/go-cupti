//go:generate enumer -type=CUpti_ApiCallbackSite -json
//go:generate enumer -type=CUpti_CallbackDomain -json
//go:generate enumer -type=CUpti_CallbackIdResource -json
//go:generate enumer -type=CUpti_CallbackIdSync -json

package types

type CUpti_ApiCallbackSite int

const (
	/**
	 * The callback is at the entry of the API call.
	 */
	CUPTI_API_ENTER CUpti_ApiCallbackSite = 0
	/**
	 * The callback is at the exit of the API call.
	 */
	CUPTI_API_EXIT             CUpti_ApiCallbackSite = 1
	CUPTI_API_CBSITE_FORCE_INT CUpti_ApiCallbackSite = 0x7fffffff
)

type CUpti_CallbackDomain int

const (
	/**
	 * Invalid domain.
	 */
	CUPTI_CB_DOMAIN_INVALID CUpti_CallbackDomain = 0
	/**
	 * Domain containing callback points for all driver API functions.
	 */
	CUPTI_CB_DOMAIN_DRIVER_API CUpti_CallbackDomain = 1
	/**
	 * Domain containing callback points for all runtime API
	 * functions.
	 */
	CUPTI_CB_DOMAIN_RUNTIME_API CUpti_CallbackDomain = 2
	/**
	 * Domain containing callback points for CUDA resource tracking.
	 */
	CUPTI_CB_DOMAIN_RESOURCE CUpti_CallbackDomain = 3
	/**
	 * Domain containing callback points for CUDA synchronization.
	 */
	CUPTI_CB_DOMAIN_SYNCHRONIZE CUpti_CallbackDomain = 4
	/**
	 * Domain containing callback points for NVTX API functions.
	 */
	CUPTI_CB_DOMAIN_NVTX      CUpti_CallbackDomain = 5
	CUPTI_CB_DOMAIN_SIZE      CUpti_CallbackDomain = 6
	CUPTI_CB_DOMAIN_FORCE_INT CUpti_CallbackDomain = 0x7fffffff
)

type CUpti_CallbackIdResource int

const (
	/**
	 * Invalid resource callback ID.
	 */
	CUPTI_CBID_RESOURCE_INVALID CUpti_CallbackIdResource = 0
	/**
	 * A new context has been created.
	 */
	CUPTI_CBID_RESOURCE_CONTEXT_CREATED CUpti_CallbackIdResource = 1
	/**
	 * A context is about to be destroyed.
	 */
	CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING CUpti_CallbackIdResource = 2
	/**
	 * A new stream has been created.
	 */
	CUPTI_CBID_RESOURCE_STREAM_CREATED CUpti_CallbackIdResource = 3
	/**
	 * A stream is about to be destroyed.
	 */
	CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING CUpti_CallbackIdResource = 4
	/**
	 * The driver has finished initializing.
	 */
	CUPTI_CBID_RESOURCE_CU_INIT_FINISHED CUpti_CallbackIdResource = 5
	/**
	 * A module has been loaded.
	 */
	CUPTI_CBID_RESOURCE_MODULE_LOADED CUpti_CallbackIdResource = 6
	/**
	 * A module is about to be unloaded.
	 */
	CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING CUpti_CallbackIdResource = 7
	/**
	 * The current module which is being profiled.
	 */
	CUPTI_CBID_RESOURCE_MODULE_PROFILED CUpti_CallbackIdResource = 8

	CUPTI_CBID_RESOURCE_SIZE      CUpti_CallbackIdResource = 9
	CUPTI_CBID_RESOURCE_FORCE_INT CUpti_CallbackIdResource = 0x7fffffff
)

type CUpti_CallbackIdSync int

const (
	/**
	 * Invalid synchronize callback ID.
	 */
	CUPTI_CBID_SYNCHRONIZE_INVALID CUpti_CallbackIdSync = 0
	/**
	 * Stream synchronization has completed for the stream.
	 */
	CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED CUpti_CallbackIdSync = 1
	/**
	 * Context synchronization has completed for the context.
	 */
	CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED CUpti_CallbackIdSync = 2
	CUPTI_CBID_SYNCHRONIZE_SIZE                 CUpti_CallbackIdSync = 3
	CUPTI_CBID_SYNCHRONIZE_FORCE_INT            CUpti_CallbackIdSync = 0x7fffffff
)
