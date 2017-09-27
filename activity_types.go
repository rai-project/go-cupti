//go:generate enumer -type=CUpti_ActivityKind -json
//go:generate enumer -type=CUpti_ActivityObjectKind -json
//go:generate enumer -type=CUpti_ActivityOverheadKind -json

package cupti

type CUpti_ActivityKind int32

const (
	/**
	 * The activity record is invalid.
	 */
	CUPTI_ACTIVITY_KIND_INVALID CUpti_ActivityKind = 0
	/**
	 * A host<->host host<->device or device<->device memory copy. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityMemcpy.
	 */
	CUPTI_ACTIVITY_KIND_MEMCPY CUpti_ActivityKind = 1
	/**
	 * A memory set executing on the GPU. The corresponding activity
	 * record structure is \ref CUpti_ActivityMemset.
	 */
	CUPTI_ACTIVITY_KIND_MEMSET CUpti_ActivityKind = 2
	/**
	 * A kernel executing on the GPU. The corresponding activity record
	 * structure is \ref CUpti_ActivityKernel4.
	 */
	CUPTI_ACTIVITY_KIND_KERNEL CUpti_ActivityKind = 3
	/**
	 * A CUDA driver API function execution. The corresponding activity
	 * record structure is \ref CUpti_ActivityAPI.
	 */
	CUPTI_ACTIVITY_KIND_DRIVER CUpti_ActivityKind = 4
	/**
	 * A CUDA runtime API function execution. The corresponding activity
	 * record structure is \ref CUpti_ActivityAPI.
	 */
	CUPTI_ACTIVITY_KIND_RUNTIME CUpti_ActivityKind = 5
	/**
	 * An event value. The corresponding activity record structure is
	 * \ref CUpti_ActivityEvent.
	 */
	CUPTI_ACTIVITY_KIND_EVENT CUpti_ActivityKind = 6
	/**
	 * A metric value. The corresponding activity record structure is
	 * \ref CUpti_ActivityMetric.
	 */
	CUPTI_ACTIVITY_KIND_METRIC CUpti_ActivityKind = 7
	/**
	 * Information about a device. The corresponding activity record
	 * structure is \ref CUpti_ActivityDevice2.
	 */
	CUPTI_ACTIVITY_KIND_DEVICE CUpti_ActivityKind = 8
	/**
	 * Information about a context. The corresponding activity record
	 * structure is \ref CUpti_ActivityContext.
	 */
	CUPTI_ACTIVITY_KIND_CONTEXT CUpti_ActivityKind = 9
	/**
	 * A (potentially concurrent) kernel executing on the GPU. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityKernel4.
	 */
	CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL CUpti_ActivityKind = 10
	/**
	 * Thread device context etc. name. The corresponding activity
	 * record structure is \ref CUpti_ActivityName.
	 */
	CUPTI_ACTIVITY_KIND_NAME CUpti_ActivityKind = 11
	/**
	 * Instantaneous start or end marker. The corresponding activity
	 * record structure is \ref CUpti_ActivityMarker2.
	 */
	CUPTI_ACTIVITY_KIND_MARKER CUpti_ActivityKind = 12
	/**
	 * Extended optional data about a marker. The corresponding
	 * activity record structure is \ref CUpti_ActivityMarkerData.
	 */
	CUPTI_ACTIVITY_KIND_MARKER_DATA CUpti_ActivityKind = 13
	/**
	 * Source information about source level result. The corresponding
	 * activity record structure is \ref CUpti_ActivitySourceLocator.
	 */
	CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR CUpti_ActivityKind = 14
	/**
	 * Results for source-level global acccess. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityGlobalAccess3.
	 */
	CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS CUpti_ActivityKind = 15
	/**
	 * Results for source-level branch. The corresponding
	 * activity record structure is \ref CUpti_ActivityBranch2.
	 */
	CUPTI_ACTIVITY_KIND_BRANCH CUpti_ActivityKind = 16
	/**
	 * Overhead activity records. The
	 * corresponding activity record structure is
	 * \ref CUpti_ActivityOverhead.
	 */
	CUPTI_ACTIVITY_KIND_OVERHEAD CUpti_ActivityKind = 17
	/**
	 * A CDP (CUDA Dynamic Parallel) kernel executing on the GPU. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityCdpKernel.  This activity can not be directly
	 * enabled or disabled. It is enabled and disabled through
	 * concurrent kernel activity i.e. _CONCURRENT_KERNEL
	 */
	CUPTI_ACTIVITY_KIND_CDP_KERNEL CUpti_ActivityKind = 18
	/**
	 * Preemption activity record indicating a preemption of a CDP (CUDA
	 * Dynamic Parallel) kernel executing on the GPU. The corresponding
	 * activity record structure is \ref CUpti_ActivityPreemption.
	 */
	CUPTI_ACTIVITY_KIND_PREEMPTION CUpti_ActivityKind = 19
	/**
	 * Environment activity records indicating power clock thermal
	 * etc. levels of the GPU. The corresponding activity record
	 * structure is \ref CUpti_ActivityEnvironment.
	 */
	CUPTI_ACTIVITY_KIND_ENVIRONMENT CUpti_ActivityKind = 20
	/**
	 * An event value associated with a specific event domain
	 * instance. The corresponding activity record structure is \ref
	 * CUpti_ActivityEventInstance.
	 */
	CUPTI_ACTIVITY_KIND_EVENT_INSTANCE CUpti_ActivityKind = 21
	/**
	 * A peer to peer memory copy. The corresponding activity record
	 * structure is \ref CUpti_ActivityMemcpy2.
	 */
	CUPTI_ACTIVITY_KIND_MEMCPY2 CUpti_ActivityKind = 22
	/**
	 * A metric value associated with a specific metric domain
	 * instance. The corresponding activity record structure is \ref
	 * CUpti_ActivityMetricInstance.
	 */
	CUPTI_ACTIVITY_KIND_METRIC_INSTANCE CUpti_ActivityKind = 23
	/**
	 * Results for source-level instruction execution.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstructionExecution.
	 */
	CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION CUpti_ActivityKind = 24
	/**
	 * Unified Memory counter record. The corresponding activity
	 * record structure is \ref CUpti_ActivityUnifiedMemoryCounter2.
	 */
	CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER CUpti_ActivityKind = 25
	/**
	 * Device global/function record. The corresponding activity
	 * record structure is \ref CUpti_ActivityFunction.
	 */
	CUPTI_ACTIVITY_KIND_FUNCTION CUpti_ActivityKind = 26
	/**
	 * CUDA Module record. The corresponding activity
	 * record structure is \ref CUpti_ActivityModule.
	 */
	CUPTI_ACTIVITY_KIND_MODULE CUpti_ActivityKind = 27
	/**
	 * A device attribute value. The corresponding activity record
	 * structure is \ref CUpti_ActivityDeviceAttribute.
	 */
	CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE CUpti_ActivityKind = 28
	/**
	 * Results for source-level shared acccess. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivitySharedAccess.
	 */
	CUPTI_ACTIVITY_KIND_SHARED_ACCESS CUpti_ActivityKind = 29
	/**
	 * Enable PC sampling for kernels. This will serialize
	 * kernels. The corresponding activity record structure
	 * is \ref CUpti_ActivityPCSampling3.
	 */
	CUPTI_ACTIVITY_KIND_PC_SAMPLING CUpti_ActivityKind = 30
	/**
	 * Summary information about PC sampling records. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityPCSamplingRecordInfo.
	 */
	CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO CUpti_ActivityKind = 31
	/**
	 * SASS/Source line-by-line correlation record.
	 * This will generate sass/source correlation for functions that have source
	 * level analysis or pc sampling results. The records will be generated only
	 * when either of source level analysis or pc sampling activity is enabled.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstructionCorrelation.
	 */
	CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION CUpti_ActivityKind = 32
	/**
	 * OpenACC data events.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityOpenAccData.
	 */
	CUPTI_ACTIVITY_KIND_OPENACC_DATA CUpti_ActivityKind = 33
	/**
	 * OpenACC launch events.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityOpenAccLaunch.
	 */
	CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH CUpti_ActivityKind = 34
	/**
	 * OpenACC other events.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityOpenAccOther.
	 */
	CUPTI_ACTIVITY_KIND_OPENACC_OTHER CUpti_ActivityKind = 35
	/**
	 * Information about a CUDA event. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityCudaEvent.
	 */
	CUPTI_ACTIVITY_KIND_CUDA_EVENT CUpti_ActivityKind = 36
	/**
	 * Information about a CUDA stream. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityStream.
	 */
	CUPTI_ACTIVITY_KIND_STREAM CUpti_ActivityKind = 37
	/**
	 * Records for synchronization management. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivitySynchronization.
	 */
	CUPTI_ACTIVITY_KIND_SYNCHRONIZATION CUpti_ActivityKind = 38
	/**
	 * Records for correlation of different programming APIs. The
	 * corresponding activity record structure is \ref
	 * CUpti_ActivityExternalCorrelation.
	 */
	CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION CUpti_ActivityKind = 39
	/**
	 * NVLink information.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityNvLink2.
	 */
	CUPTI_ACTIVITY_KIND_NVLINK CUpti_ActivityKind = 40
	/**
	 * Instantaneous Event information.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstantaneousEvent.
	 */
	CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT CUpti_ActivityKind = 41
	/**
	 * Instantaneous Event information for a specific event
	 * domain instance.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstantaneousEventInstance
	 */
	CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE CUpti_ActivityKind = 42
	/**
	 * Instantaneous Metric information
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstantaneousMetric.
	 */
	CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC CUpti_ActivityKind = 43
	/**
	 * Instantaneous Metric information for a specific metric
	 * domain instance.
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityInstantaneousMetricInstance.
	 */
	CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE CUpti_ActivityKind = 44
	/*
	 * Memory activity tracking allocation and freeing of the memory
	 * The corresponding activity record structure is \ref
	 * CUpti_ActivityMemory.
	 */
	CUPTI_ACTIVITY_KIND_MEMORY CUpti_ActivityKind = 45

	CUPTI_ACTIVITY_KIND_FORCE_INT CUpti_ActivityKind = 0x7fffffff
)

type CUpti_ActivityObjectKind int32

const (
	/**
	 * The object kind is not known.
	 */
	CUPTI_ACTIVITY_OBJECT_UNKNOWN CUpti_ActivityObjectKind = 0
	/**
	 * A process.
	 */
	CUPTI_ACTIVITY_OBJECT_PROCESS CUpti_ActivityObjectKind = 1
	/**
	 * A thread.
	 */
	CUPTI_ACTIVITY_OBJECT_THREAD CUpti_ActivityObjectKind = 2
	/**
	 * A device.
	 */
	CUPTI_ACTIVITY_OBJECT_DEVICE CUpti_ActivityObjectKind = 3
	/**
	 * A context.
	 */
	CUPTI_ACTIVITY_OBJECT_CONTEXT CUpti_ActivityObjectKind = 4
	/**
	 * A stream.
	 */
	CUPTI_ACTIVITY_OBJECT_STREAM CUpti_ActivityObjectKind = 5

	CUPTI_ACTIVITY_OBJECT_FORCE_INT CUpti_ActivityObjectKind = 0x7fffffff
)

type CUpti_ActivityOverheadKind int32

const (
	/**
	 * The overhead kind is not known.
	 */
	CUPTI_ACTIVITY_OVERHEAD_UNKNOWN CUpti_ActivityOverheadKind = 0
	/**
	 * Compiler(JIT) overhead.
	 */
	CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER CUpti_ActivityOverheadKind = 1
	/**
	 * Activity buffer flush overhead.
	 */
	CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH CUpti_ActivityOverheadKind = 1 << 16
	/**
	 * CUPTI instrumentation overhead.
	 */
	CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION CUpti_ActivityOverheadKind = 2 << 16
	/**
	 * CUPTI resource creation and destruction overhead.
	 */
	CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE CUpti_ActivityOverheadKind = 3 << 16
	CUPTI_ACTIVITY_OVERHEAD_FORCE_INT      CUpti_ActivityOverheadKind = 0x7fffffff
)
