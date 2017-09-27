//go:generate enumer -type=CUpti_ActivityKind -json
//go:generate enumer -type=CUpti_ActivityObjectKind -json
//go:generate enumer -type=CUpti_ActivityOverheadKind -json
//go:generate enumer -type=CUpti_ActivityComputeApiKind -json
//go:generate enumer -type=CUpti_ActivityFlag -json
//go:generate enumer -type=CUpti_ActivityPCSamplingStallReason -json
//go:generate enumer -type=CUpti_ActivityPCSamplingPeriod -json
//go:generate enumer -type=CUpti_ActivityMemcpyKind -json
//go:generate enumer -type=CUpti_ActivityMemoryKind -json
//go:generate enumer -type=CUpti_ActivityPreemptionKind -json
//go:generate enumer -type=CUpti_ActivityEnvironmentKind -json
//go:generate enumer -type=CUpti_EnvironmentClocksThrottleReason -json
//go:generate enumer -type=CUpti_ActivityUnifiedMemoryCounterScope -json
//go:generate enumer -type=CUpti_ActivityUnifiedMemoryCounterKind -json
//go:generate enumer -type=CUpti_ActivityUnifiedMemoryAccessType -json
//go:generate enumer -type=CUpti_ActivityUnifiedMemoryMigrationCause -json
//go:generate enumer -type=CUpti_ActivityUnifiedMemoryRemoteMapCause -json
//go:generate enumer -type=CUpti_ActivityInstructionClass -json
//go:generate enumer -type=CUpti_ActivityPartitionedGlobalCacheConfig -json
//go:generate enumer -type=CUpti_ActivitySynchronizationType -json
//go:generate enumer -type=CUpti_ActivityStreamFlag -json
//go:generate enumer -type=CUpti_LinkFlag -json
//go:generate enumer -type=CUpti_DevType -json
//go:generate enumer -type=CUpti_ActivityAttribute -json
//go:generate enumer -type=CUpti_ActivityThreadIdType -json

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

type CUpti_ActivityComputeApiKind int

const (
	/**
	 * The compute API is not known.
	 */
	CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN CUpti_ActivityComputeApiKind = 0
	/**
	 * The compute APIs are for CUDA.
	 */
	CUPTI_ACTIVITY_COMPUTE_API_CUDA CUpti_ActivityComputeApiKind = 1
	/**
	 * The compute APIs are for CUDA running
	 * in MPS (Multi-Process Service) environment.
	 */
	CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS CUpti_ActivityComputeApiKind = 2

	CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT CUpti_ActivityComputeApiKind = 0x7fffffff
)

type CUpti_ActivityFlag int

const (
	/**
	 * Indicates the activity record has no flags.
	 */
	CUPTI_ACTIVITY_FLAG_NONE CUpti_ActivityFlag = 0

	/**
	 * Indicates the activity represents a device that supports
	 * concurrent kernel execution. Valid for
	 * CUPTI_ACTIVITY_KIND_DEVICE.
	 */
	CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates if the activity represents a CUdevice_attribute value
	 * or a CUpti_DeviceAttribute value. Valid for
	 * CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
	 */
	CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents an asynchronous memcpy
	 * operation. Valid for CUPTI_ACTIVITY_KIND_MEMCPY.
	 */
	CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents an instantaneous marker. Valid
	 * for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents a region start marker. Valid
	 * for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_START CUpti_ActivityFlag = 1 << 1

	/**
	 * Indicates the activity represents a region end marker. Valid for
	 * CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_END CUpti_ActivityFlag = 1 << 2

	/**
	 * Indicates the activity represents an attempt to acquire a user
	 * defined synchronization object.
	 * Valid for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE CUpti_ActivityFlag = 1 << 3

	/**
	 * Indicates the activity represents success in acquiring the
	 * user defined synchronization object.
	 * Valid for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS CUpti_ActivityFlag = 1 << 4

	/**
	 * Indicates the activity represents failure in acquiring the
	 * user defined synchronization object.
	 * Valid for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED CUpti_ActivityFlag = 1 << 5

	/**
	 * Indicates the activity represents releasing a reservation on
	 * user defined synchronization object.
	 * Valid for CUPTI_ACTIVITY_KIND_MARKER.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE CUpti_ActivityFlag = 1 << 6

	/**
	 * Indicates the activity represents a marker that does not specify
	 * a color. Valid for CUPTI_ACTIVITY_KIND_MARKER_DATA.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents a marker that specifies a color
	 * in alpha-red-green-blue format. Valid for
	 * CUPTI_ACTIVITY_KIND_MARKER_DATA.
	 */
	CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB CUpti_ActivityFlag = 1 << 1

	/**
	 * The number of bytes requested by each thread
	 * Valid for CUpti_ActivityGlobalAccess3.
	 */
	CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK CUpti_ActivityFlag = 0xFF << 0
	/**
	 * If bit in this flag is set the access was load else it is a
	 * store access. Valid for CUpti_ActivityGlobalAccess3.
	 */
	CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD CUpti_ActivityFlag = 1 << 8
	/**
	 * If this bit in flag is set the load access was cached else it is
	 * uncached. Valid for CUpti_ActivityGlobalAccess3.
	 */
	CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED CUpti_ActivityFlag = 1 << 9
	/**
	 * If this bit in flag is set the metric value overflowed. Valid
	 * for CUpti_ActivityMetric and CUpti_ActivityMetricInstance.
	 */
	CUPTI_ACTIVITY_FLAG_METRIC_OVERFLOWED CUpti_ActivityFlag = 1 << 0
	/**
	 * If this bit in flag is set the metric value couldn't be
	 * calculated. This occurs when a value(s) required to calculate the
	 * metric is missing.  Valid for CUpti_ActivityMetric and
	 * CUpti_ActivityMetricInstance.
	 */
	CUPTI_ACTIVITY_FLAG_METRIC_VALUE_INVALID CUpti_ActivityFlag = 1 << 1
	/**
	 * If this bit in flag is set the source level metric value couldn't be
	 * calculated. This occurs when a value(s) required to calculate the
	 * source level metric cannot be evaluated.
	 * Valid for CUpti_ActivityInstructionExecution.
	 */
	CUPTI_ACTIVITY_FLAG_INSTRUCTION_VALUE_INVALID CUpti_ActivityFlag = 1 << 0
	/**
	 * The mask for the instruction class \ref CUpti_ActivityInstructionClass
	 * Valid for CUpti_ActivityInstructionExecution and
	 * CUpti_ActivityInstructionCorrelation
	 */
	CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK CUpti_ActivityFlag = 0xFF << 1
	/**
	 * When calling cuptiActivityFlushAll this flag
	 * can be set to force CUPTI to flush all records in the buffer whether
	 * finished or not
	 */
	CUPTI_ACTIVITY_FLAG_FLUSH_FORCED CUpti_ActivityFlag = 1 << 0

	/**
	 * The number of bytes requested by each thread
	 * Valid for CUpti_ActivitySharedAccess.
	 */
	CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_SIZE_MASK CUpti_ActivityFlag = 0xFF << 0
	/**
	 * If bit in this flag is set the access was load else it is a
	 * store access.  Valid for CUpti_ActivitySharedAccess.
	 */
	CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_LOAD CUpti_ActivityFlag = 1 << 8

	/**
	 * Indicates the activity represents an asynchronous memset
	 * operation. Valid for CUPTI_ACTIVITY_KIND_MEMSET.
	 */
	CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents thrashing in CPU.
	 * Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING in
	 * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
	 */
	CUPTI_ACTIVITY_FLAG_THRASHING_IN_CPU CUpti_ActivityFlag = 1 << 0

	/**
	 * Indicates the activity represents page throttling in CPU.
	 * Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING in
	 * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
	 */
	CUPTI_ACTIVITY_FLAG_THROTTLING_IN_CPU CUpti_ActivityFlag = 1 << 0

	CUPTI_ACTIVITY_FLAG_FORCE_INT CUpti_ActivityFlag = 0x7fffffff
)

type CUpti_ActivityPCSamplingStallReason int

const (
	/**
	 * Invalid reason
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID CUpti_ActivityPCSamplingStallReason = 0
	/**
	 * No stall instruction is selected for issue
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE CUpti_ActivityPCSamplingStallReason = 1
	/**
	 * Warp is blocked because next instruction is not yet available
	 * because of instruction cache miss or because of branching effects
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH CUpti_ActivityPCSamplingStallReason = 2
	/**
	 * Instruction is waiting on an arithmatic dependency
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY CUpti_ActivityPCSamplingStallReason = 3
	/**
	 * Warp is blocked because it is waiting for a memory access to complete.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY CUpti_ActivityPCSamplingStallReason = 4
	/**
	 * Texture sub-system is fully utilized or has too many outstanding requests.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE CUpti_ActivityPCSamplingStallReason = 5
	/**
	 * Warp is blocked as it is waiting at __syncthreads() or at memory barrier.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC CUpti_ActivityPCSamplingStallReason = 6
	/**
	 * Warp is blocked waiting for __constant__ memory and immediate memory access to complete.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY CUpti_ActivityPCSamplingStallReason = 7
	/**
	 * Compute operation cannot be performed due to the required resources not
	 * being available.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY CUpti_ActivityPCSamplingStallReason = 8
	/**
	 * Warp is blocked because there are too many pending memory operations.
	 * In Kepler architecture it often indicates high number of memory replays.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE CUpti_ActivityPCSamplingStallReason = 9
	/**
	 * Warp was ready to issue but some other warp issued instead.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED CUpti_ActivityPCSamplingStallReason = 10
	/**
	 * Miscellaneous reasons
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER CUpti_ActivityPCSamplingStallReason = 11
	/**
	 * Sleeping.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING  CUpti_ActivityPCSamplingStallReason = 12
	CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT CUpti_ActivityPCSamplingStallReason = 0x7fffffff
)

type CUpti_ActivityPCSamplingPeriod int

const (
	/**
	 * The PC sampling period is not set.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID CUpti_ActivityPCSamplingPeriod = 0
	/**
	 * Minimum sampling period available on the device.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN CUpti_ActivityPCSamplingPeriod = 1
	/**
	 * Sampling period in lower range.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW CUpti_ActivityPCSamplingPeriod = 2
	/**
	 * Medium sampling period.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID CUpti_ActivityPCSamplingPeriod = 3
	/**
	 * Sampling period in higher range.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH CUpti_ActivityPCSamplingPeriod = 4
	/**
	 * Maximum sampling period available on the device.
	 */
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX       CUpti_ActivityPCSamplingPeriod = 5
	CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT CUpti_ActivityPCSamplingPeriod = 0x7fffffff
)

type CUpti_ActivityMemcpyKind int

const (
	/**
	 * The memory copy kind is not known.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN CUpti_ActivityMemcpyKind = 0
	/**
	 * A host to device memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_HTOD CUpti_ActivityMemcpyKind = 1
	/**
	 * A device to host memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_DTOH CUpti_ActivityMemcpyKind = 2
	/**
	 * A host to device array memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_HTOA CUpti_ActivityMemcpyKind = 3
	/**
	 * A device array to host memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_ATOH CUpti_ActivityMemcpyKind = 4
	/**
	 * A device array to device array memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_ATOA CUpti_ActivityMemcpyKind = 5
	/**
	 * A device array to device memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_ATOD CUpti_ActivityMemcpyKind = 6
	/**
	 * A device to device array memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_DTOA CUpti_ActivityMemcpyKind = 7
	/**
	 * A device to device memory copy on the same device.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_DTOD CUpti_ActivityMemcpyKind = 8
	/**
	 * A host to host memory copy.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_HTOH CUpti_ActivityMemcpyKind = 9
	/**
	 * A peer to peer memory copy across different devices.
	 */
	CUPTI_ACTIVITY_MEMCPY_KIND_PTOP CUpti_ActivityMemcpyKind = 10

	CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT CUpti_ActivityMemcpyKind = 0x7fffffff
)

type CUpti_ActivityMemoryKind int

const (
	/**
	 * The memory kind is unknown.
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN CUpti_ActivityMemoryKind = 0
	/**
	 * The memory is pageable.
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE CUpti_ActivityMemoryKind = 1
	/**
	 * The memory is pinned.
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_PINNED CUpti_ActivityMemoryKind = 2
	/**
	 * The memory is on the device.
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_DEVICE CUpti_ActivityMemoryKind = 3
	/**
	 * The memory is an array.
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_ARRAY CUpti_ActivityMemoryKind = 4
	/**
	 * The memory is managed
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_MANAGED CUpti_ActivityMemoryKind = 5
	/**
	 * The memory is device static
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC CUpti_ActivityMemoryKind = 6
	/**
	 * The memory is managed static
	 */
	CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC CUpti_ActivityMemoryKind = 7
	CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT      CUpti_ActivityMemoryKind = 0x7fffffff
)

type CUpti_ActivityPreemptionKind int

const (
	/**
	 * The preemption kind is not known.
	 */
	CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN CUpti_ActivityPreemptionKind = 0
	/**
	 * Preemption to save CDP block.
	 */
	CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE CUpti_ActivityPreemptionKind = 1
	/**
	 * Preemption to restore CDP block.
	 */
	CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE   CUpti_ActivityPreemptionKind = 2
	CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT CUpti_ActivityPreemptionKind = 0x7fffffff
)

type CUpti_ActivityEnvironmentKind int

const (
	/**
	 * Unknown data.
	 */
	CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN CUpti_ActivityEnvironmentKind = 0
	/**
	 * The environment data is related to speed.
	 */
	CUPTI_ACTIVITY_ENVIRONMENT_SPEED CUpti_ActivityEnvironmentKind = 1
	/**
	 * The environment data is related to temperature.
	 */
	CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE CUpti_ActivityEnvironmentKind = 2
	/**
	 * The environment data is related to power.
	 */
	CUPTI_ACTIVITY_ENVIRONMENT_POWER CUpti_ActivityEnvironmentKind = 3
	/**
	 * The environment data is related to cooling.
	 */
	CUPTI_ACTIVITY_ENVIRONMENT_COOLING CUpti_ActivityEnvironmentKind = 4

	CUPTI_ACTIVITY_ENVIRONMENT_COUNT
	CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT CUpti_ActivityEnvironmentKind = 0x7fffffff
)

type CUpti_EnvironmentClocksThrottleReason int

const (
	/**
	 * Nothing is running on the GPU and the clocks are dropping to idle
	 * state.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE CUpti_EnvironmentClocksThrottleReason = 0x00000001
	/**
	 * The GPU clocks are limited by a user specified limit.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS CUpti_EnvironmentClocksThrottleReason = 0x00000002
	/**
	 * A software power scaling algorithm is reducing the clocks below
	 * requested clocks.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP CUpti_EnvironmentClocksThrottleReason = 0x00000004
	/**
	 * Hardware slowdown to reduce the clock by a factor of two or more
	 * is engaged.  This is an indicator of one of the following: 1)
	 * Temperature is too high 2) External power brake assertion is
	 * being triggered (e.g. by the system power supply) 3) Change in
	 * power state.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN CUpti_EnvironmentClocksThrottleReason = 0x00000008
	/**
	 * Some unspecified factor is reducing the clocks.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_UNKNOWN CUpti_EnvironmentClocksThrottleReason = 0x80000000
	/**
	 * Throttle reason is not supported for this GPU.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED CUpti_EnvironmentClocksThrottleReason = 0x40000000
	/**
	 * No clock throttling.
	 */
	CUPTI_CLOCKS_THROTTLE_REASON_NONE CUpti_EnvironmentClocksThrottleReason = 0x00000000

	CUPTI_CLOCKS_THROTTLE_REASON_FORCE_INT CUpti_EnvironmentClocksThrottleReason = 0x7fffffff
)

type CUpti_ActivityUnifiedMemoryCounterScope int

const (
	/**
	 * The unified memory counter scope is not known.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN CUpti_ActivityUnifiedMemoryCounterScope = 0
	/**
	 * Collect unified memory counter for single process on one device
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE CUpti_ActivityUnifiedMemoryCounterScope = 1
	/**
	 * Collect unified memory counter for single process across all devices
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES CUpti_ActivityUnifiedMemoryCounterScope = 2

	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT CUpti_ActivityUnifiedMemoryCounterScope = 0x7fffffff
)

type CUpti_ActivityUnifiedMemoryCounterKind int

const (
	/**
	 * The unified memory counter kind is not known.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN CUpti_ActivityUnifiedMemoryCounterKind = 0
	/**
	 * Number of bytes transfered from host to device
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD CUpti_ActivityUnifiedMemoryCounterKind = 1
	/**
	 * Number of bytes transfered from device to host
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH CUpti_ActivityUnifiedMemoryCounterKind = 2
	/**
	 * Number of CPU page faults this is only supported on 64 bit
	 * Linux and Mac platforms
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT CUpti_ActivityUnifiedMemoryCounterKind = 3
	/**
	 * Number of GPU page faults this is only supported on devices with
	 * compute capability 6.0 and higher and 64 bit Linux platforms
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT CUpti_ActivityUnifiedMemoryCounterKind = 4
	/**
	 * Thrashing occurs when data is frequently accessed by
	 * multiple processors and has to be constantly migrated around
	 * to achieve data locality. In this case the overhead of migration
	 * may exceed the benefits of locality.
	 * This is only supported on 64 bit Linux platforms.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING CUpti_ActivityUnifiedMemoryCounterKind = 5
	/**
	 * Throttling is a prevention technique used by the driver to avoid
	 * further thrashing. Here the driver doesn't service the fault for
	 * one of the contending processors for a specific period of time
	 * so that the other processor can run at full-speed.
	 * This is only supported on 64 bit Linux platforms.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING CUpti_ActivityUnifiedMemoryCounterKind = 6
	/**
	 * In case throttling does not help the driver tries to pin the memory
	 * to a processor for a specific period of time. One of the contending
	 * processors will have slow  access to the memory while the other will
	 * have fast access.
	 * This is only supported on 64 bit Linux platforms.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP CUpti_ActivityUnifiedMemoryCounterKind = 7

	/**
	 * Number of bytes transferred from one device to another device.
	 * This is only supported on 64 bit Linux platforms.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD CUpti_ActivityUnifiedMemoryCounterKind = 8

	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT
	CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT CUpti_ActivityUnifiedMemoryCounterKind = 0x7fffffff
)

type CUpti_ActivityUnifiedMemoryAccessType int

const (
	/**
	 * The unified memory access type is not known
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN CUpti_ActivityUnifiedMemoryAccessType = 0
	/**
	 * The page fault was triggered by read memory instruction
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ CUpti_ActivityUnifiedMemoryAccessType = 1
	/**
	 * The page fault was triggered by write memory instruction
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE CUpti_ActivityUnifiedMemoryAccessType = 2
	/**
	 * The page fault was triggered by atomic memory instruction
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC CUpti_ActivityUnifiedMemoryAccessType = 3
	/**
	 * The page fault was triggered by memory prefetch operation
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH CUpti_ActivityUnifiedMemoryAccessType = 4
)

type CUpti_ActivityUnifiedMemoryMigrationCause int

const (
	/**
	 * The unified memory migration cause is not known
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN CUpti_ActivityUnifiedMemoryMigrationCause = 0
	/**
	 * The unified memory migrated due to an explicit call from
	 * the user e.g. cudaMemPrefetchAsync
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER CUpti_ActivityUnifiedMemoryMigrationCause = 1
	/**
	 * The unified memory migrated to guarantee data coherence
	 * e.g. CPU/GPU faults on Pascal+ and kernel launch on pre-Pascal GPUs
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE CUpti_ActivityUnifiedMemoryMigrationCause = 2
	/**
	 * The unified memory was speculatively migrated by the UVM driver
	 * before being accessed by the destination processor to improve
	 * performance
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH CUpti_ActivityUnifiedMemoryMigrationCause = 3
	/**
	 * The unified memory migrated to the CPU because it was evicted to make
	 * room for another block of memory on the GPU
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION CUpti_ActivityUnifiedMemoryMigrationCause = 4
)

type CUpti_ActivityUnifiedMemoryRemoteMapCause int

const (
	/**
	 * The cause of mapping to remote memory was unknown
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN CUpti_ActivityUnifiedMemoryRemoteMapCause = 0
	/**
	 * Mapping to remote memory was added to maintain data coherence.
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE CUpti_ActivityUnifiedMemoryRemoteMapCause = 1
	/**
	 * Mapping to remote memory was added to prevent further thrashing
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING CUpti_ActivityUnifiedMemoryRemoteMapCause = 2
	/**
	 * Mapping to remote memory was added to enforce the hints
	 * specified by the programmer or by performance heuristics of the
	 * UVM driver
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY CUpti_ActivityUnifiedMemoryRemoteMapCause = 3
	/**
	 * Mapping to remote memory was added because there is no more
	 * memory available on the processor and eviction was not
	 * possible
	 */
	CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY CUpti_ActivityUnifiedMemoryRemoteMapCause = 4
)

type CUpti_ActivityInstructionClass int

const (
	/**
	 * The instruction class is not known.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN CUpti_ActivityInstructionClass = 0
	/**
	 * Represents a 32 bit floating point operation.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32 CUpti_ActivityInstructionClass = 1
	/**
	 * Represents a 64 bit floating point operation.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64 CUpti_ActivityInstructionClass = 2
	/**
	 * Represents an integer operation.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER CUpti_ActivityInstructionClass = 3
	/**
	 * Represents a bit conversion operation.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION CUpti_ActivityInstructionClass = 4
	/**
	 * Represents a control flow instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW CUpti_ActivityInstructionClass = 5
	/**
	 * Represents a global load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL CUpti_ActivityInstructionClass = 6
	/**
	 * Represents a shared load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED CUpti_ActivityInstructionClass = 7
	/**
	 * Represents a local load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL CUpti_ActivityInstructionClass = 8
	/**
	 * Represents a generic load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC CUpti_ActivityInstructionClass = 9
	/**
	 * Represents a surface load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE CUpti_ActivityInstructionClass = 10
	/**
	 * Represents a constant load instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT CUpti_ActivityInstructionClass = 11
	/**
	 * Represents a texture load-store instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE CUpti_ActivityInstructionClass = 12
	/**
	 * Represents a global atomic instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC CUpti_ActivityInstructionClass = 13
	/**
	 * Represents a shared atomic instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC CUpti_ActivityInstructionClass = 14
	/**
	 * Represents a surface atomic instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC CUpti_ActivityInstructionClass = 15
	/**
	 * Represents a inter-thread communication instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION CUpti_ActivityInstructionClass = 16
	/**
	 * Represents a barrier instruction.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER CUpti_ActivityInstructionClass = 17
	/**
	 * Represents some miscellaneous instructions which do not fit in the above classification.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS CUpti_ActivityInstructionClass = 18
	/**
	 * Represents a 16 bit floating point operation.
	 */
	CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16 CUpti_ActivityInstructionClass = 19

	CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT CUpti_ActivityInstructionClass = 0x7fffffff
)

type CUpti_ActivityPartitionedGlobalCacheConfig int

const (
	/**
	 * Partitioned global cache config unknown.
	 */
	CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN CUpti_ActivityPartitionedGlobalCacheConfig = 0
	/**
	 * Partitioned global cache not supported.
	 */
	CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED CUpti_ActivityPartitionedGlobalCacheConfig = 1
	/**
	 * Partitioned global cache config off.
	 */
	CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF CUpti_ActivityPartitionedGlobalCacheConfig = 2
	/**
	 * Partitioned global cache config on.
	 */
	CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON        CUpti_ActivityPartitionedGlobalCacheConfig = 3
	CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT CUpti_ActivityPartitionedGlobalCacheConfig = 0x7fffffff
)

type CUpti_ActivitySynchronizationType int

const (
	/**
	 * Unknown data.
	 */
	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN CUpti_ActivitySynchronizationType = 0
	/**
	 * Event synchronize API.
	 */
	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE CUpti_ActivitySynchronizationType = 1
	/**
	 * Stream wait event API.
	 */
	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT CUpti_ActivitySynchronizationType = 2
	/**
	 * Stream synchronize API.
	 */
	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE CUpti_ActivitySynchronizationType = 3
	/**
	 * Context synchronize API.
	 */
	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE CUpti_ActivitySynchronizationType = 4

	CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT CUpti_ActivitySynchronizationType = 0x7fffffff
)

type CUpti_ActivityStreamFlag int

const (
	/**
	 * Unknown data.
	 */
	CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN CUpti_ActivityStreamFlag = 0
	/**
	 * Default stream.
	 */
	CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT CUpti_ActivityStreamFlag = 1
	/**
	 * Non-blocking stream.
	 */
	CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING CUpti_ActivityStreamFlag = 2
	/**
	 * Null stream.
	 */
	CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL CUpti_ActivityStreamFlag = 3
	/**
	 * Stream create Mask
	 */
	CUPTI_ACTIVITY_STREAM_CREATE_MASK CUpti_ActivityStreamFlag = 0xFFFF

	CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT CUpti_ActivityStreamFlag = 0x7fffffff
)

type CUpti_LinkFlag int

const (
	CUPTI_LINK_FLAG_INVALID CUpti_LinkFlag = 0
	/**
	 * Is peer to peer access supported by this link.
	 */
	CUPTI_LINK_FLAG_PEER_ACCESS CUpti_LinkFlag = (1 << 1)
	/**
	 * Is system memory access supported by this link.
	 */
	CUPTI_LINK_FLAG_SYSMEM_ACCESS CUpti_LinkFlag = (1 << 2)
	/**
	 * Is peer atomic access supported by this link.
	 */
	CUPTI_LINK_FLAG_PEER_ATOMICS CUpti_LinkFlag = (1 << 3)
	/**
	 * Is system memory atomic access supported by this link.
	 */
	CUPTI_LINK_FLAG_SYSMEM_ATOMICS CUpti_LinkFlag = (1 << 4)

	CUPTI_LINK_FLAG_FORCE_INT CUpti_LinkFlag = 0x7fffffff
)

type CUpti_DevType int

const (
	CUPTI_DEV_TYPE_INVALID CUpti_DevType = 0
	/**
	 * The device type is GPU.
	 */
	CUPTI_DEV_TYPE_GPU CUpti_DevType = 1
	/**
	 * The device type is NVLink processing unit in CPU.
	 */
	CUPTI_DEV_TYPE_NPU       CUpti_DevType = 2
	CUPTI_DEV_TYPE_FORCE_INT CUpti_DevType = 0x7fffffff
)

type CUpti_ActivityAttribute int

const (
	/**
	 * The device memory size (in bytes) reserved for storing profiling
	 * data for non-CDP operations especially for concurrent kernel
	 * tracing for each buffer on a context. The value is a size_t.
	 *
	 * Having larger buffer size means less flush operations but
	 * consumes more device memory. Having smaller buffer size
	 * increases the risk of dropping timestamps for kernel records
	 * if too many kernels are launched/replayed at one time. This
	 * value only applies to new buffer allocations.
	 *
	 * Set this value before initializing CUDA or before creating a
	 * context to ensure it is considered for the following allocations.
	 *
	 * The default value is 8388608 (8MB).
	 *
	 * Note: The actual amount of device memory per buffer reserved by
	 * CUPTI might be larger.
	 */
	CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE CUpti_ActivityAttribute = 0
	/**
	 * The device memory size (in bytes) reserved for storing profiling
	 * data for CDP operations for each buffer on a context. The
	 * value is a size_t.
	 *
	 * Having larger buffer size means less flush operations but
	 * consumes more device memory. This value only applies to new
	 * allocations.
	 *
	 * Set this value before initializing CUDA or before creating a
	 * context to ensure it is considered for the following allocations.
	 *
	 * The default value is 8388608 (8MB).
	 *
	 * Note: The actual amount of device memory per context reserved by
	 * CUPTI might be larger.
	 */
	CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP CUpti_ActivityAttribute = 1
	/**
	 * The maximum number of memory buffers per context. The value is
	 * a size_t.
	 *
	 * Buffers can be reused by the context. Increasing this value
	 * reduces the number of times CUPTI needs to flush the buffers.
	 * Setting this value will not modify the number of memory buffers
	 * currently stored.
	 *
	 * Set this value before initializing CUDA to ensure the limit is
	 * not exceeded.
	 *
	 * The default value is 100.
	 */
	CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT CUpti_ActivityAttribute = 2

	/**
	 * The profiling semaphore pool size reserved for storing profiling
	 * data for serialized kernels and memory operations for each context.
	 * The value is a size_t.
	 *
	 * Having larger pool size means less semaphore query operations but
	 * consumes more device resources. Having smaller pool size increases
	 * the risk of dropping timestamps for kernel and memcpy records if
	 * too many kernels or memcpy are launched/replayed at one time.
	 * This value only applies to new pool allocations.
	 *
	 * Set this value before initializing CUDA or before creating a
	 * context to ensure it is considered for the following allocations.
	 *
	 * The default value is 65536.
	 *
	 */
	CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE CUpti_ActivityAttribute = 3
	/**
	 * The maximum number of profiling semaphore pools per context. The
	 * value is a size_t.
	 *
	 * Profiling semaphore pool can be reused by the context. Increasing
	 * this value reduces the number of times CUPTI needs to query semaphores
	 * in the pool. Setting this value will not modify the number of
	 * semaphore pools currently stored.
	 *
	 * Set this value before initializing CUDA to ensure the limit is
	 * not exceeded.
	 *
	 * The default value is 100.
	 */
	CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT CUpti_ActivityAttribute = 4

	CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT CUpti_ActivityAttribute = 0x7fffffff
)

type CUpti_ActivityThreadIdType int

const (
	/**
	 * Default type
	 * Windows uses API GetCurrentThreadId()
	 * Linux/Mac/Android/QNX use POSIX pthread API pthread_self()
	 */
	CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT CUpti_ActivityThreadIdType = 0

	/**
	 * This type is based on the system API available on the underlying platform
	 * and thread-id obtained is supposed to be unique for the process lifetime.
	 * Windows uses API GetCurrentThreadId()
	 * Linux uses syscall SYS_gettid
	 * Mac uses syscall SYS_thread_selfid
	 * Android/QNX use gettid()
	 */
	CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM CUpti_ActivityThreadIdType = 1

	CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT CUpti_ActivityThreadIdType = 0x7fffffff
)
