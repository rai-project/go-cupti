//go:generate enumer -type=CUdevice_attribute -json
//go:generate enumer -type=CUresult -json

package types

type CUdevice_attribute int

const (
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                   CUdevice_attribute = 1  /**< Maximum number of threads per block */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                         CUdevice_attribute = 2  /**< Maximum block dimension X */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                         CUdevice_attribute = 3  /**< Maximum block dimension Y */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                         CUdevice_attribute = 4  /**< Maximum block dimension Z */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                          CUdevice_attribute = 5  /**< Maximum grid dimension X */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                          CUdevice_attribute = 6  /**< Maximum grid dimension Y */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                          CUdevice_attribute = 7  /**< Maximum grid dimension Z */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK             CUdevice_attribute = 8  /**< Maximum shared memory available per block in bytes */
	CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                 CUdevice_attribute = 8  /**< Deprecated use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                   CUdevice_attribute = 9  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
	CU_DEVICE_ATTRIBUTE_WARP_SIZE                               CUdevice_attribute = 10 /**< Warp size in threads */
	CU_DEVICE_ATTRIBUTE_MAX_PITCH                               CUdevice_attribute = 11 /**< Maximum pitch in bytes allowed by memory copies */
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                 CUdevice_attribute = 12 /**< Maximum number of 32-bit registers available per block */
	CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                     CUdevice_attribute = 12 /**< Deprecated use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE                              CUdevice_attribute = 13 /**< Typical clock frequency in kilohertz */
	CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                       CUdevice_attribute = 14 /**< Alignment requirement for textures */
	CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                             CUdevice_attribute = 15 /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                    CUdevice_attribute = 16 /**< Number of multiprocessors on device */
	CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                     CUdevice_attribute = 17 /**< Specifies whether there is a run time limit on kernels */
	CU_DEVICE_ATTRIBUTE_INTEGRATED                              CUdevice_attribute = 18 /**< Device is integrated with host memory */
	CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                     CUdevice_attribute = 19 /**< Device can map host memory into CUDA address space */
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                            CUdevice_attribute = 20 /**< Compute mode (See ::CUcomputemode for details) */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                 CUdevice_attribute = 21 /**< Maximum 1D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                 CUdevice_attribute = 22 /**< Maximum 2D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                CUdevice_attribute = 23 /**< Maximum 2D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                 CUdevice_attribute = 24 /**< Maximum 3D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                CUdevice_attribute = 25 /**< Maximum 3D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                 CUdevice_attribute = 26 /**< Maximum 3D texture depth */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH         CUdevice_attribute = 27 /**< Maximum 2D layered texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT        CUdevice_attribute = 28 /**< Maximum 2D layered texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS        CUdevice_attribute = 29 /**< Maximum layers in a 2D layered texture */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH           CUdevice_attribute = 27 /**< Deprecated use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT          CUdevice_attribute = 28 /**< Deprecated use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES       CUdevice_attribute = 29 /**< Deprecated use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
	CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                       CUdevice_attribute = 30 /**< Alignment requirement for surfaces */
	CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                      CUdevice_attribute = 31 /**< Device can possibly execute multiple kernels concurrently */
	CU_DEVICE_ATTRIBUTE_ECC_ENABLED                             CUdevice_attribute = 32 /**< Device has ECC support enabled */
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                              CUdevice_attribute = 33 /**< PCI bus ID of the device */
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                           CUdevice_attribute = 34 /**< PCI device ID of the device */
	CU_DEVICE_ATTRIBUTE_TCC_DRIVER                              CUdevice_attribute = 35 /**< Device is using TCC driver model */
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                       CUdevice_attribute = 36 /**< Peak memory clock frequency in kilohertz */
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                 CUdevice_attribute = 37 /**< Global memory bus width in bits */
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                           CUdevice_attribute = 38 /**< Size of L2 cache in bytes */
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR          CUdevice_attribute = 39 /**< Maximum resident threads per multiprocessor */
	CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                      CUdevice_attribute = 40 /**< Number of asynchronous engines */
	CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                      CUdevice_attribute = 41 /**< Device shares a unified address space with the host */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH         CUdevice_attribute = 42 /**< Maximum 1D layered texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS        CUdevice_attribute = 43 /**< Maximum layers in a 1D layered texture */
	CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                        CUdevice_attribute = 44 /**< Deprecated do not use. */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH          CUdevice_attribute = 45 /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT         CUdevice_attribute = 46 /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE       CUdevice_attribute = 47 /**< Alternate maximum 3D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE      CUdevice_attribute = 48 /**< Alternate maximum 3D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE       CUdevice_attribute = 49 /**< Alternate maximum 3D texture depth */
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                           CUdevice_attribute = 50 /**< PCI domain ID of the device */
	CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                 CUdevice_attribute = 51 /**< Pitch alignment requirement for textures */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH            CUdevice_attribute = 52 /**< Maximum cubemap texture width/height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH    CUdevice_attribute = 53 /**< Maximum cubemap layered texture width/height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS   CUdevice_attribute = 54 /**< Maximum layers in a cubemap layered texture */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                 CUdevice_attribute = 55 /**< Maximum 1D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                 CUdevice_attribute = 56 /**< Maximum 2D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                CUdevice_attribute = 57 /**< Maximum 2D surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                 CUdevice_attribute = 58 /**< Maximum 3D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                CUdevice_attribute = 59 /**< Maximum 3D surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                 CUdevice_attribute = 60 /**< Maximum 3D surface depth */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH         CUdevice_attribute = 61 /**< Maximum 1D layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS        CUdevice_attribute = 62 /**< Maximum layers in a 1D layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH         CUdevice_attribute = 63 /**< Maximum 2D layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT        CUdevice_attribute = 64 /**< Maximum 2D layered surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS        CUdevice_attribute = 65 /**< Maximum layers in a 2D layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH            CUdevice_attribute = 66 /**< Maximum cubemap surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH    CUdevice_attribute = 67 /**< Maximum cubemap layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS   CUdevice_attribute = 68 /**< Maximum layers in a cubemap layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH          CUdevice_attribute = 69 /**< Maximum 1D linear texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH          CUdevice_attribute = 70 /**< Maximum 2D linear texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT         CUdevice_attribute = 71 /**< Maximum 2D linear texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH          CUdevice_attribute = 72 /**< Maximum 2D linear texture pitch in bytes */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH       CUdevice_attribute = 73 /**< Maximum mipmapped 2D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT      CUdevice_attribute = 74 /**< Maximum mipmapped 2D texture height */
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                CUdevice_attribute = 75 /**< Major compute capability version number */
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                CUdevice_attribute = 76 /**< Minor compute capability version number */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH       CUdevice_attribute = 77 /**< Maximum mipmapped 1D texture width */
	CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED             CUdevice_attribute = 78 /**< Device supports stream priorities */
	CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED               CUdevice_attribute = 79 /**< Device supports caching globals in L1 */
	CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                CUdevice_attribute = 80 /**< Device supports caching locals in L1 */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR    CUdevice_attribute = 81 /**< Maximum shared memory available per multiprocessor in bytes */
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR        CUdevice_attribute = 82 /**< Maximum number of 32-bit registers available per multiprocessor */
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                          CUdevice_attribute = 83 /**< Device can allocate managed memory on this system */
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                         CUdevice_attribute = 84 /**< Device is on a multi-GPU board */
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                CUdevice_attribute = 85 /**< Unique id for a group of devices on the same multi-GPU board */
	CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED            CUdevice_attribute = 86 /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute and is not supported on any current hardware)*/
	CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO   CUdevice_attribute = 87 /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                  CUdevice_attribute = 88 /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS               CUdevice_attribute = 89 /**< Device can coherently access managed memory concurrently with the CPU */
	CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED            CUdevice_attribute = 90 /**< Device supports compute preemption. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM CUdevice_attribute = 91 /**< Device can access host registered memory at the same virtual address as the CPU */
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS                  CUdevice_attribute = 92 /**< ::cuStreamBatchMemOp and related APIs are supported. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS           CUdevice_attribute = 93 /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR           CUdevice_attribute = 94 /**< ::CU_STREAM_WAIT_VALUE_NOR is supported. */
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                      CUdevice_attribute = 95 /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH         CUdevice_attribute = 96 /**< Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN       CUdevice_attribute = 97 /**< Maximum optin shared memory per block */
	CU_DEVICE_ATTRIBUTE_MAX
)

type CUresult int

const (
	/**
	 * The API call returned with no errors. In the case of query calls this
	 * can also mean that the operation being queried is complete (see
	 * ::cuEventQuery() and ::cuStreamQuery()).
	 */
	CUDA_SUCCESS CUresult = 0

	/**
	 * This indicates that one or more of the parameters passed to the API call
	 * is not within an acceptable range of values.
	 */
	CUDA_ERROR_INVALID_VALUE CUresult = 1

	/**
	 * The API call failed because it was unable to allocate enough memory to
	 * perform the requested operation.
	 */
	CUDA_ERROR_OUT_OF_MEMORY CUresult = 2

	/**
	 * This indicates that the CUDA driver has not been initialized with
	 * ::cuInit() or that initialization has failed.
	 */
	CUDA_ERROR_NOT_INITIALIZED CUresult = 3

	/**
	 * This indicates that the CUDA driver is in the process of shutting down.
	 */
	CUDA_ERROR_DEINITIALIZED CUresult = 4

	/**
	 * This indicates profiler is not initialized for this run. This can
	 * happen when the application is running with external profiling tools
	 * like visual profiler.
	 */
	CUDA_ERROR_PROFILER_DISABLED CUresult = 5

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to attempt to enable/disable the profiling via ::cuProfilerStart or
	 * ::cuProfilerStop without initialization.
	 */
	CUDA_ERROR_PROFILER_NOT_INITIALIZED CUresult = 6

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call cuProfilerStart() when profiling is already enabled.
	 */
	CUDA_ERROR_PROFILER_ALREADY_STARTED CUresult = 7

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call cuProfilerStop() when profiling is already disabled.
	 */
	CUDA_ERROR_PROFILER_ALREADY_STOPPED CUresult = 8

	/**
	 * This indicates that no CUDA-capable devices were detected by the installed
	 * CUDA driver.
	 */
	CUDA_ERROR_NO_DEVICE CUresult = 100

	/**
	 * This indicates that the device ordinal supplied by the user does not
	 * correspond to a valid CUDA device.
	 */
	CUDA_ERROR_INVALID_DEVICE CUresult = 101

	/**
	 * This indicates that the device kernel image is invalid. This can also
	 * indicate an invalid CUDA module.
	 */
	CUDA_ERROR_INVALID_IMAGE CUresult = 200

	/**
	 * This most frequently indicates that there is no context bound to the
	 * current thread. This can also be returned if the context passed to an
	 * API call is not a valid handle (such as a context that has had
	 * ::cuCtxDestroy() invoked on it). This can also be returned if a user
	 * mixes different API versions (i.e. 3010 context with 3020 API calls).
	 * See ::cuCtxGetApiVersion() for more details.
	 */
	CUDA_ERROR_INVALID_CONTEXT CUresult = 201

	/**
	 * This indicated that the context being supplied as a parameter to the
	 * API call was already the active context.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.2. It is no longer an
	 * error to attempt to push the active context via ::cuCtxPushCurrent().
	 */
	CUDA_ERROR_CONTEXT_ALREADY_CURRENT CUresult = 202

	/**
	 * This indicates that a map or register operation has failed.
	 */
	CUDA_ERROR_MAP_FAILED CUresult = 205

	/**
	 * This indicates that an unmap or unregister operation has failed.
	 */
	CUDA_ERROR_UNMAP_FAILED CUresult = 206

	/**
	 * This indicates that the specified array is currently mapped and thus
	 * cannot be destroyed.
	 */
	CUDA_ERROR_ARRAY_IS_MAPPED CUresult = 207

	/**
	 * This indicates that the resource is already mapped.
	 */
	CUDA_ERROR_ALREADY_MAPPED CUresult = 208

	/**
	 * This indicates that there is no kernel image available that is suitable
	 * for the device. This can occur when a user specifies code generation
	 * options for a particular CUDA source file that do not include the
	 * corresponding device configuration.
	 */
	CUDA_ERROR_NO_BINARY_FOR_GPU CUresult = 209

	/**
	 * This indicates that a resource has already been acquired.
	 */
	CUDA_ERROR_ALREADY_ACQUIRED CUresult = 210

	/**
	 * This indicates that a resource is not mapped.
	 */
	CUDA_ERROR_NOT_MAPPED CUresult = 211

	/**
	 * This indicates that a mapped resource is not available for access as an
	 * array.
	 */
	CUDA_ERROR_NOT_MAPPED_AS_ARRAY CUresult = 212

	/**
	 * This indicates that a mapped resource is not available for access as a
	 * pointer.
	 */
	CUDA_ERROR_NOT_MAPPED_AS_POINTER CUresult = 213

	/**
	 * This indicates that an uncorrectable ECC error was detected during
	 * execution.
	 */
	CUDA_ERROR_ECC_UNCORRECTABLE CUresult = 214

	/**
	 * This indicates that the ::CUlimit passed to the API call is not
	 * supported by the active device.
	 */
	CUDA_ERROR_UNSUPPORTED_LIMIT CUresult = 215

	/**
	 * This indicates that the ::CUcontext passed to the API call can
	 * only be bound to a single CPU thread at a time but is already
	 * bound to a CPU thread.
	 */
	CUDA_ERROR_CONTEXT_ALREADY_IN_USE CUresult = 216

	/**
	 * This indicates that peer access is not supported across the given
	 * devices.
	 */
	CUDA_ERROR_PEER_ACCESS_UNSUPPORTED CUresult = 217

	/**
	 * This indicates that a PTX JIT compilation failed.
	 */
	CUDA_ERROR_INVALID_PTX CUresult = 218

	/**
	 * This indicates an error with OpenGL or DirectX context.
	 */
	CUDA_ERROR_INVALID_GRAPHICS_CONTEXT CUresult = 219

	/**
	 * This indicates that an uncorrectable NVLink error was detected during the
	 * execution.
	 */
	CUDA_ERROR_NVLINK_UNCORRECTABLE CUresult = 220

	/**
	 * This indicates that the PTX JIT compiler library was not found.
	 */
	CUDA_ERROR_JIT_COMPILER_NOT_FOUND CUresult = 221

	/**
	 * This indicates that the device kernel source is invalid.
	 */
	CUDA_ERROR_INVALID_SOURCE CUresult = 300

	/**
	 * This indicates that the file specified was not found.
	 */
	CUDA_ERROR_FILE_NOT_FOUND CUresult = 301

	/**
	 * This indicates that a link to a shared object failed to resolve.
	 */
	CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND CUresult = 302

	/**
	 * This indicates that initialization of a shared object failed.
	 */
	CUDA_ERROR_SHARED_OBJECT_INIT_FAILED CUresult = 303

	/**
	 * This indicates that an OS call failed.
	 */
	CUDA_ERROR_OPERATING_SYSTEM CUresult = 304

	/**
	 * This indicates that a resource handle passed to the API call was not
	 * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
	 */
	CUDA_ERROR_INVALID_HANDLE CUresult = 400

	/**
	 * This indicates that a named symbol was not found. Examples of symbols
	 * are global/constant variable names texture names and surface names.
	 */
	CUDA_ERROR_NOT_FOUND CUresult = 500

	/**
	 * This indicates that asynchronous operations issued previously have not
	 * completed yet. This result is not actually an error but must be indicated
	 * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
	 * may return this value include ::cuEventQuery() and ::cuStreamQuery().
	 */
	CUDA_ERROR_NOT_READY CUresult = 600

	/**
	 * While executing a kernel the device encountered a
	 * load or store instruction on an invalid memory address.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_ILLEGAL_ADDRESS CUresult = 700

	/**
	 * This indicates that a launch did not occur because it did not have
	 * appropriate resources. This error usually indicates that the user has
	 * attempted to pass too many arguments to the device kernel or the
	 * kernel launch specifies too many threads for the kernel's register
	 * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
	 * when a 32-bit int is expected) is equivalent to passing too many
	 * arguments and can also result in this error.
	 */
	CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES CUresult = 701

	/**
	 * This indicates that the device kernel took too long to execute. This can
	 * only occur if timeouts are enabled - see the device attribute
	 * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_LAUNCH_TIMEOUT CUresult = 702

	/**
	 * This error indicates a kernel launch that uses an incompatible texturing
	 * mode.
	 */
	CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING CUresult = 703

	/**
	 * This error indicates that a call to ::cuCtxEnablePeerAccess() is
	 * trying to re-enable peer access to a context which has already
	 * had peer access to it enabled.
	 */
	CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED CUresult = 704

	/**
	 * This error indicates that ::cuCtxDisablePeerAccess() is
	 * trying to disable peer access which has not been enabled yet
	 * via ::cuCtxEnablePeerAccess().
	 */
	CUDA_ERROR_PEER_ACCESS_NOT_ENABLED CUresult = 705

	/**
	 * This error indicates that the primary context for the specified device
	 * has already been initialized.
	 */
	CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE CUresult = 708

	/**
	 * This error indicates that the context current to the calling thread
	 * has been destroyed using ::cuCtxDestroy or is a primary context which
	 * has not yet been initialized.
	 */
	CUDA_ERROR_CONTEXT_IS_DESTROYED CUresult = 709

	/**
	 * A device-side assert triggered during kernel execution. The context
	 * cannot be used anymore and must be destroyed. All existing device
	 * memory allocations from this context are invalid and must be
	 * reconstructed if the program is to continue using CUDA.
	 */
	CUDA_ERROR_ASSERT CUresult = 710

	/**
	 * This error indicates that the hardware resources required to enable
	 * peer access have been exhausted for one or more of the devices
	 * passed to ::cuCtxEnablePeerAccess().
	 */
	CUDA_ERROR_TOO_MANY_PEERS CUresult = 711

	/**
	 * This error indicates that the memory range passed to ::cuMemHostRegister()
	 * has already been registered.
	 */
	CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED CUresult = 712

	/**
	 * This error indicates that the pointer passed to ::cuMemHostUnregister()
	 * does not correspond to any currently registered memory region.
	 */
	CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED CUresult = 713

	/**
	 * While executing a kernel the device encountered a stack error.
	 * This can be due to stack corruption or exceeding the stack size limit.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_HARDWARE_STACK_ERROR CUresult = 714

	/**
	 * While executing a kernel the device encountered an illegal instruction.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_ILLEGAL_INSTRUCTION CUresult = 715

	/**
	 * While executing a kernel the device encountered a load or store instruction
	 * on a memory address which is not aligned.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_MISALIGNED_ADDRESS CUresult = 716

	/**
	 * While executing a kernel the device encountered an instruction
	 * which can only operate on memory locations in certain address spaces
	 * (global shared or local) but was supplied a memory address not
	 * belonging to an allowed address space.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_INVALID_ADDRESS_SPACE CUresult = 717

	/**
	 * While executing a kernel the device program counter wrapped its address space.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_INVALID_PC CUresult = 718

	/**
	 * An exception occurred on the device while executing a kernel. Common
	 * causes include dereferencing an invalid device pointer and accessing
	 * out of bounds shared memory.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDA_ERROR_LAUNCH_FAILED CUresult = 719

	/**
	 * This error indicates that the number of blocks launched per grid for a kernel that was
	 * launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
	 * exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
	 * or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
	 * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
	 */
	CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE CUresult = 720

	/**
	 * This error indicates that the attempted operation is not permitted.
	 */
	CUDA_ERROR_NOT_PERMITTED CUresult = 800

	/**
	 * This error indicates that the attempted operation is not supported
	 * on the current system or device.
	 */
	CUDA_ERROR_NOT_SUPPORTED CUresult = 801

	/**
	 * This indicates that an unknown internal error has occurred.
	 */
	CUDA_ERROR_UNKNOWN CUresult = 999
)
