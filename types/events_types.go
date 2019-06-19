package types

type CUpti_DeviceAttributeDeviceClass int

const (
	CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA   CUpti_DeviceAttributeDeviceClass = 0
	CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO  CUpti_DeviceAttributeDeviceClass = 1
	CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE CUpti_DeviceAttributeDeviceClass = 2
	CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA   CUpti_DeviceAttributeDeviceClass = 3
)

type CUpti_DeviceAttribute int

const (
	/**
	 * Number of event IDs for a device. Value is a uint32_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_EVENT_ID CUpti_DeviceAttribute = 1
	/**
	 * Number of event domain IDs for a device. Value is a uint32_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID CUpti_DeviceAttribute = 2
	/**
	 * Get global memory bandwidth in Kbytes/sec. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH CUpti_DeviceAttribute = 3
	/**
	 * Get theoretical maximum number of instructions per cycle. Value
	 * is a uint32_t.
	 */
	CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE CUpti_DeviceAttribute = 4
	/**
	 * Get theoretical maximum number of single precision instructions
	 * that can be executed per second. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION CUpti_DeviceAttribute = 5
	/**
	 * Get number of frame buffers for device.  Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS CUpti_DeviceAttribute = 6
	/**
	 * Get PCIE link rate in Mega bits/sec for device. Return 0 if bus-type
	 * is non-PCIE. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_PCIE_LINK_RATE CUpti_DeviceAttribute = 7
	/**
	 * Get PCIE link width for device. Return 0 if bus-type
	 * is non-PCIE. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH CUpti_DeviceAttribute = 8
	/**
	 * Get PCIE generation for device. Return 0 if bus-type
	 * is non-PCIE. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_PCIE_GEN CUpti_DeviceAttribute = 9
	/**
	 * Get the class for the device. Value is a
	 * CUpti_DeviceAttributeDeviceClass.
	 */
	CUPTI_DEVICE_ATTR_DEVICE_CLASS CUpti_DeviceAttribute = 10
	/**
	 * Get the peak single precision flop per cycle. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE CUpti_DeviceAttribute = 11
	/**
	 * Get the peak double precision flop per cycle. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE CUpti_DeviceAttribute = 12
	/**
	 * Get number of L2 units. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_L2_UNITS CUpti_DeviceAttribute = 13
	/**
	 * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_SHARED
	 * preference. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED CUpti_DeviceAttribute = 14
	/**
	 * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_L1
	 * preference. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 CUpti_DeviceAttribute = 15
	/**
	 * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_EQUAL
	 * preference. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL CUpti_DeviceAttribute = 16
	/**
	 * Get the peak half precision flop per cycle. Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE CUpti_DeviceAttribute = 17
	/**
	 * Check if Nvlink is connected to device. Returns 1 if at least one
	 * Nvlink is connected to the device returns 0 otherwise.
	 * Value is a uint32_t.
	 */
	CUPTI_DEVICE_ATTR_NVLINK_PRESENT CUpti_DeviceAttribute = 18
	/**
	 * Check if Nvlink is present between GPU and CPU. Returns Bandwidth
	 * in Bytes/sec if Nvlink is present returns 0 otherwise.
	 * Value is a uint64_t.
	 */
	CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW CUpti_DeviceAttribute = 19
	CUPTI_DEVICE_ATTR_FORCE_INT         CUpti_DeviceAttribute = 0x7fffffff
)

type CUpti_EventDomainAttribute int

const (
	/**
	 * Event domain name. Value is a null terminated const c-string.
	 */
	CUPTI_EVENT_DOMAIN_ATTR_NAME CUpti_EventDomainAttribute = 0
	/**
	 * Number of instances of the domain for which event counts will be
	 * collected.  The domain may have additional instances that cannot
	 * be profiled (see CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT).
	 * Can be read only with \ref
	 * cuptiDeviceGetEventDomainAttribute. Value is a uint32_t.
	 */
	CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT CUpti_EventDomainAttribute = 1
	/**
	 * Total number of instances of the domain including instances that
	 * cannot be profiled.  Use CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT
	 * to get the number of instances that can be profiled. Can be read
	 * only with \ref cuptiDeviceGetEventDomainAttribute. Value is a
	 * uint32_t.
	 */
	CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT CUpti_EventDomainAttribute = 3
	/**
	 * Collection method used for events contained in the event domain.
	 * Value is a \ref CUpti_EventCollectionMethod.
	 */
	CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD CUpti_EventDomainAttribute = 4

	CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT CUpti_EventDomainAttribute = 0x7fffffff
)

type CUpti_EventCollectionMethod int

const (
	/**
	 * Event is collected using a hardware global performance monitor.
	 */
	CUPTI_EVENT_COLLECTION_METHOD_PM CUpti_EventCollectionMethod = 0
	/**
	 * Event is collected using a hardware SM performance monitor.
	 */
	CUPTI_EVENT_COLLECTION_METHOD_SM CUpti_EventCollectionMethod = 1
	/**
	 * Event is collected using software instrumentation.
	 */
	CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED CUpti_EventCollectionMethod = 2
	/**
	 * Event is collected using NvLink throughput counter method.
	 */
	CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC CUpti_EventCollectionMethod = 3
	CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT CUpti_EventCollectionMethod = 0x7fffffff
)

type CUpti_EventGroupAttribute int

const (
	/**
	 * The domain to which the event group is bound. This attribute is
	 * set when the first event is added to the group.  Value is a
	 * CUpti_EventDomainID.
	 */
	CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID CUpti_EventGroupAttribute = 0
	/**
	 * [rw] Profile all the instances of the domain for this
	 * eventgroup. This feature can be used to get load balancing
	 * across all instances of a domain. Value is an integer.
	 */
	CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES CUpti_EventGroupAttribute = 1
	/**
	 * [rw] Reserved for user data.
	 */
	CUPTI_EVENT_GROUP_ATTR_USER_DATA CUpti_EventGroupAttribute = 2
	/**
	 * Number of events in the group. Value is a uint32_t.
	 */
	CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS CUpti_EventGroupAttribute = 3
	/**
	 * Enumerates events in the group. Value is a pointer to buffer of
	 * size sizeof(CUpti_EventID) * num_of_events in the eventgroup.
	 * num_of_events can be queried using
	 * CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS.
	 */
	CUPTI_EVENT_GROUP_ATTR_EVENTS CUpti_EventGroupAttribute = 4
	/**
	 * Number of instances of the domain bound to this event group that
	 * will be counted.  Value is a uint32_t.
	 */
	CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT CUpti_EventGroupAttribute = 5
	CUPTI_EVENT_GROUP_ATTR_FORCE_INT      CUpti_EventGroupAttribute = 0x7fffffff
)

type CUpti_EventAttribute int

const (
	/**
	 * Event name. Value is a null terminated const c-string.
	 */
	CUPTI_EVENT_ATTR_NAME CUpti_EventAttribute = 0
	/**
	 * Short description of event. Value is a null terminated const
	 * c-string.
	 */
	CUPTI_EVENT_ATTR_SHORT_DESCRIPTION CUpti_EventAttribute = 1
	/**
	 * Long description of event. Value is a null terminated const
	 * c-string.
	 */
	CUPTI_EVENT_ATTR_LONG_DESCRIPTION CUpti_EventAttribute = 2
	/**
	 * Category of event. Value is CUpti_EventCategory.
	 */
	CUPTI_EVENT_ATTR_CATEGORY  CUpti_EventAttribute = 3
	CUPTI_EVENT_ATTR_FORCE_INT CUpti_EventAttribute = 0x7fffffff
)

type CUpti_EventCollectionMode int

const (
	/**
	 * Events are collected for the entire duration between the
	 * cuptiEventGroupEnable and cuptiEventGroupDisable calls.
	 * For devices with compute capability less than 2.0 event
	 * values are reset when a kernel is launched. For all other
	 * devices event values are only reset when the events are read.
	 * For CUDA toolkit v6.0 and older this was the default mode.
	 */
	CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS CUpti_EventCollectionMode = 0
	/**
	 * Events are collected only for the durations of kernel executions
	 * that occur between the cuptiEventGroupEnable and
	 * cuptiEventGroupDisable calls. Event collection begins when a
	 * kernel execution begins and stops when kernel execution
	 * completes. Event values are reset to zero when each kernel
	 * execution begins. If multiple kernel executions occur between the
	 * cuptiEventGroupEnable and cuptiEventGroupDisable calls then the
	 * event values must be read after each kernel launch if those
	 * events need to be associated with the specific kernel launch.
	 * This is the default mode from CUDA toolkit v6.5.
	 */
	CUPTI_EVENT_COLLECTION_MODE_KERNEL    CUpti_EventCollectionMode = 1
	CUPTI_EVENT_COLLECTION_MODE_FORCE_INT CUpti_EventCollectionMode = 0x7fffffff
)

type CUpti_EventCategory int

const (
	/**
	 * An instruction related event.
	 */
	CUPTI_EVENT_CATEGORY_INSTRUCTION CUpti_EventCategory = 0
	/**
	 * A memory related event.
	 */
	CUPTI_EVENT_CATEGORY_MEMORY CUpti_EventCategory = 1
	/**
	 * A cache related event.
	 */
	CUPTI_EVENT_CATEGORY_CACHE CUpti_EventCategory = 2
	/**
	 * A profile-trigger event.
	 */
	CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER CUpti_EventCategory = 3
	CUPTI_EVENT_CATEGORY_FORCE_INT       CUpti_EventCategory = 0x7fffffff
)

type CUpti_ReadEventFlags int

const (
	/**
	 * No flags.
	 */
	CUPTI_EVENT_READ_FLAG_NONE      CUpti_ReadEventFlags = 0
	CUPTI_EVENT_READ_FLAG_FORCE_INT CUpti_ReadEventFlags = 0x7fffffff
)
