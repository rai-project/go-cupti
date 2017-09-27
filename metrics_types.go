//go:generate enumer -type=CUpti_MetricCategory -json
//go:generate enumer -type=CUpti_MetricEvaluationMode -json
//go:generate enumer -type=CUpti_MetricValueKind -json
//go:generate enumer -type=CUpti_MetricValueUtilizationLevel -json
//go:generate enumer -type=CUpti_MetricAttribute -json
//go:generate enumer -type=CUpti_MetricPropertyDeviceClass -json
//go:generate enumer -type=CUpti_MetricPropertyID -json

package cupti

type CUpti_MetricCategory int

const (
	/**
	 * A memory related metric.
	 */
	CUPTI_METRIC_CATEGORY_MEMORY CUpti_MetricCategory = 0
	/**
	 * An instruction related metric.
	 */
	CUPTI_METRIC_CATEGORY_INSTRUCTION CUpti_MetricCategory = 1
	/**
	 * A multiprocessor related metric.
	 */
	CUPTI_METRIC_CATEGORY_MULTIPROCESSOR CUpti_MetricCategory = 2
	/**
	 * A cache related metric.
	 */
	CUPTI_METRIC_CATEGORY_CACHE CUpti_MetricCategory = 3
	/**
	 * A texture related metric.
	 */
	CUPTI_METRIC_CATEGORY_TEXTURE CUpti_MetricCategory = 4
	/**
	 *A Nvlink related metric.
	 */
	CUPTI_METRIC_CATEGORY_NVLINK    CUpti_MetricCategory = 5
	CUPTI_METRIC_CATEGORY_FORCE_INT CUpti_MetricCategory = 0x7fffffff
)

type CUpti_MetricEvaluationMode int

const (
	/**
	 * If this bit is set the metric can be profiled for each instance of the
	 * domain. The event values passed to \ref cuptiMetricGetValue can contain
	 * values for one instance of the domain. And \ref cuptiMetricGetValue can
	 * be called for each instance.
	 */
	CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE CUpti_MetricEvaluationMode = 1
	/**
	 * If this bit is set the metric can be profiled over all instances. The
	 * event values passed to \ref cuptiMetricGetValue can be aggregated values
	 * of events for all instances of the domain.
	 */
	CUPTI_METRIC_EVALUATION_MODE_AGGREGATE CUpti_MetricEvaluationMode = 1 << 1
	CUPTI_METRIC_EVALUATION_MODE_FORCE_INT CUpti_MetricEvaluationMode = 0x7fffffff
)

type CUpti_MetricValueKind int

const (
	/**
	 * The metric value is a 64-bit double.
	 */
	CUPTI_METRIC_VALUE_KIND_DOUBLE CUpti_MetricValueKind = 0
	/**
	 * The metric value is a 64-bit unsigned integer.
	 */
	CUPTI_METRIC_VALUE_KIND_UINT64 CUpti_MetricValueKind = 1
	/**
	 * The metric value is a percentage represented by a 64-bit
	 * double. For example 57.5% is represented by the value 57.5.
	 */
	CUPTI_METRIC_VALUE_KIND_PERCENT CUpti_MetricValueKind = 2
	/**
	 * The metric value is a throughput represented by a 64-bit
	 * integer. The unit for throughput values is bytes/second.
	 */
	CUPTI_METRIC_VALUE_KIND_THROUGHPUT CUpti_MetricValueKind = 3
	/**
	 * The metric value is a 64-bit signed integer.
	 */
	CUPTI_METRIC_VALUE_KIND_INT64 CUpti_MetricValueKind = 4
	/**
	 * The metric value is a utilization level as represented by
	 * CUpti_MetricValueUtilizationLevel.
	 */
	CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL CUpti_MetricValueKind = 5

	CUPTI_METRIC_VALUE_KIND_FORCE_INT CUpti_MetricValueKind = 0x7fffffff
)

type CUpti_MetricValueUtilizationLevel int

const (
	CUPTI_METRIC_VALUE_UTILIZATION_IDLE      CUpti_MetricValueUtilizationLevel = 0
	CUPTI_METRIC_VALUE_UTILIZATION_LOW       CUpti_MetricValueUtilizationLevel = 2
	CUPTI_METRIC_VALUE_UTILIZATION_MID       CUpti_MetricValueUtilizationLevel = 5
	CUPTI_METRIC_VALUE_UTILIZATION_HIGH      CUpti_MetricValueUtilizationLevel = 8
	CUPTI_METRIC_VALUE_UTILIZATION_MAX       CUpti_MetricValueUtilizationLevel = 10
	CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT CUpti_MetricValueUtilizationLevel = 0x7fffffff
)

type CUpti_MetricAttribute int

const (
	/**
	 * Metric name. Value is a null terminated const c-string.
	 */
	CUPTI_METRIC_ATTR_NAME CUpti_MetricAttribute = 0
	/**
	 * Short description of metric. Value is a null terminated const c-string.
	 */
	CUPTI_METRIC_ATTR_SHORT_DESCRIPTION CUpti_MetricAttribute = 1
	/**
	 * Long description of metric. Value is a null terminated const c-string.
	 */
	CUPTI_METRIC_ATTR_LONG_DESCRIPTION CUpti_MetricAttribute = 2
	/**
	 * Category of the metric. Value is of type CUpti_MetricCategory.
	 */
	CUPTI_METRIC_ATTR_CATEGORY CUpti_MetricAttribute = 3
	/**
	 * Value type of the metric. Value is of type CUpti_MetricValueKind.
	 */
	CUPTI_METRIC_ATTR_VALUE_KIND CUpti_MetricAttribute = 4
	/**
	 * Metric evaluation mode. Value is of type CUpti_MetricEvaluationMode.
	 */
	CUPTI_METRIC_ATTR_EVALUATION_MODE CUpti_MetricAttribute = 5
	CUPTI_METRIC_ATTR_FORCE_INT       CUpti_MetricAttribute = 0x7fffffff
)

type CUpti_MetricPropertyDeviceClass int

const (
	CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA   CUpti_MetricPropertyDeviceClass = 0
	CUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO  CUpti_MetricPropertyDeviceClass = 1
	CUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE CUpti_MetricPropertyDeviceClass = 2
	CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA   CUpti_MetricPropertyDeviceClass = 3
)

type CUpti_MetricPropertyID int

const (
	/*
	 * Number of multiprocessors on a device.  This can be collected
	 * using value of \param CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT of
	 * cuDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT CUpti_MetricPropertyID = 0
	/*
	 * Maximum number of warps on a multiprocessor. This can be
	 * collected using ratio of value of \param
	 * CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR and \param
	 * CU_DEVICE_ATTRIBUTE_WARP_SIZE of cuDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR
	/*
	 * GPU Time for kernel in ns. This should be profiled using CUPTI
	 * Activity API.
	 */
	CUPTI_METRIC_PROPERTY_KERNEL_GPU_TIME
	/*
	 * Clock rate for device in KHz.  This should be collected using
	 * value of \param CU_DEVICE_ATTRIBUTE_CLOCK_RATE of
	 * cuDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_CLOCK_RATE
	/*
	 * Number of Frame buffer units for device. This should be collected
	 * using value of \param CUPTI_DEVICE_ATTRIBUTE_MAX_FRAME_BUFFERS of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT
	/*
	 * Global memory bandwidth in KBytes/sec. This should be collected
	 * using value of \param CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH
	 * of cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH
	/*
	 * PCIE link rate in Mega bits/sec. This should be collected using
	 * value of \param CUPTI_DEVICE_ATTR_PCIE_LINK_RATE of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_PCIE_LINK_RATE
	/*
	 * PCIE link width for device. This should be collected using
	 * value of \param CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH
	/*
	 * PCIE generation for device. This should be collected using
	 * value of \param CUPTI_DEVICE_ATTR_PCIE_GEN of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_PCIE_GEN
	/*
	 * The device class. This should be collected using
	 * value of \param CUPTI_DEVICE_ATTR_DEVICE_CLASS of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_DEVICE_CLASS
	/*
	 * Peak single precision floating point operations that
	 * can be performed in one cycle by the device.
	 * This should be collected using value of
	 * \param CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE
	/*
	 * Peak double precision floating point operations that
	 * can be performed in one cycle by the device.
	 * This should be collected using value of
	 * \param CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE
	/*
	 * Number of L2 units on a device. This can be collected
	 * using value of \param CUPTI_DEVICE_ATTR_MAX_L2_UNITS of
	 * cuDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_L2_UNITS
	/*
	 * Whether ECC support is enabled on the device. This can be
	 * collected using value of \param CU_DEVICE_ATTRIBUTE_ECC_ENABLED of
	 * cuDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_ECC_ENABLED
	/*
	 * Peak half precision floating point operations that
	 * can be performed in one cycle by the device.
	 * This should be collected using value of
	 * \param CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE
	/*
	 * NVLINK Bandwitdh for device. This should be collected
	 * using value of \param CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW of
	 * cuptiDeviceGetAttribute.
	 */
	CUPTI_METRIC_PROPERTY_GPU_CPU_NVLINK_BANDWIDTH
)
