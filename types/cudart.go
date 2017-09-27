//go:generate enumer -type=CUDAError -json
package types

type CUDAError int

const (
	/**
	 * The API call returned with no errors. In the case of query calls this
	 * can also mean that the operation being queried is complete (see
	 * ::CUDAEventQuery() and ::CUDAStreamQuery()).
	 */
	CUDASuccess CUDAError = 0

	/**
	 * The device function being invoked (usually via ::CUDALaunchKernel()) was not
	 * previously configured via the ::CUDAConfigureCall() function.
	 */
	CUDAErrorMissingConfiguration CUDAError = 1

	/**
	 * The API call failed because it was unable to allocate enough memory to
	 * perform the requested operation.
	 */
	CUDAErrorMemoryAllocation CUDAError = 2

	/**
	 * The API call failed because the CUDA driver and runtime could not be
	 * initialized.
	 */
	CUDAErrorInitializationError CUDAError = 3

	/**
	 * An exception occurred on the device while executing a kernel. Common
	 * causes include dereferencing an invalid device pointer and accessing
	 * out of bounds shared memory. The device cannot be used until
	 * ::CUDAThreadExit() is called. All existing device memory allocations
	 * are invalid and must be reconstructed if the program is to continue
	 * using CUDA.
	 */
	CUDAErrorLaunchFailure CUDAError = 4

	/**
	 * This indicated that a previous kernel launch failed. This was previously
	 * used for device emulation of kernel launches.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorPriorLaunchFailure CUDAError = 5

	/**
	 * This indicates that the device kernel took too long to execute. This can
	 * only occur if timeouts are enabled - see the device property
	 * \ref ::CUDADeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
	 * for more information.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorLaunchTimeout CUDAError = 6

	/**
	 * This indicates that a launch did not occur because it did not have
	 * appropriate resources. Although this error is similar to
	 * ::CUDAErrorInvalidConfiguration this error usually indicates that the
	 * user has attempted to pass too many arguments to the device kernel or the
	 * kernel launch specifies too many threads for the kernel's register count.
	 */
	CUDAErrorLaunchOutOfResources CUDAError = 7

	/**
	 * The requested device function does not exist or is not compiled for the
	 * proper device architecture.
	 */
	CUDAErrorInvalidDeviceFunction CUDAError = 8

	/**
	 * This indicates that a kernel launch is requesting resources that can
	 * never be satisfied by the current device. Requesting more shared memory
	 * per block than the device supports will trigger this error as will
	 * requesting too many threads or blocks. See ::CUDADeviceProp for more
	 * device limitations.
	 */
	CUDAErrorInvalidConfiguration CUDAError = 9

	/**
	 * This indicates that the device ordinal supplied by the user does not
	 * correspond to a valid CUDA device.
	 */
	CUDAErrorInvalidDevice CUDAError = 10

	/**
	 * This indicates that one or more of the parameters passed to the API call
	 * is not within an acceptable range of values.
	 */
	CUDAErrorInvalidValue CUDAError = 11

	/**
	 * This indicates that one or more of the pitch-related parameters passed
	 * to the API call is not within the acceptable range for pitch.
	 */
	CUDAErrorInvalidPitchValue CUDAError = 12

	/**
	 * This indicates that the symbol name/identifier passed to the API call
	 * is not a valid name or identifier.
	 */
	CUDAErrorInvalidSymbol CUDAError = 13

	/**
	 * This indicates that the buffer object could not be mapped.
	 */
	CUDAErrorMapBufferObjectFailed CUDAError = 14

	/**
	 * This indicates that the buffer object could not be unmapped.
	 */
	CUDAErrorUnmapBufferObjectFailed CUDAError = 15

	/**
	 * This indicates that at least one host pointer passed to the API call is
	 * not a valid host pointer.
	 */
	CUDAErrorInvalidHostPointer CUDAError = 16

	/**
	 * This indicates that at least one device pointer passed to the API call is
	 * not a valid device pointer.
	 */
	CUDAErrorInvalidDevicePointer CUDAError = 17

	/**
	 * This indicates that the texture passed to the API call is not a valid
	 * texture.
	 */
	CUDAErrorInvalidTexture CUDAError = 18

	/**
	 * This indicates that the texture binding is not valid. This occurs if you
	 * call ::CUDAGetTextureAlignmentOffset() with an unbound texture.
	 */
	CUDAErrorInvalidTextureBinding CUDAError = 19

	/**
	 * This indicates that the channel descriptor passed to the API call is not
	 * valid. This occurs if the format is not one of the formats specified by
	 * ::CUDAChannelFormatKind or if one of the dimensions is invalid.
	 */
	CUDAErrorInvalidChannelDescriptor CUDAError = 20

	/**
	 * This indicates that the direction of the memcpy passed to the API call is
	 * not one of the types specified by ::CUDAMemcpyKind.
	 */
	CUDAErrorInvalidMemcpyDirection CUDAError = 21

	/**
	 * This indicated that the user has taken the address of a constant variable
	 * which was forbidden up until the CUDA 3.1 release.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Variables in constant
	 * memory may now have their address taken by the runtime via
	 * ::CUDAGetSymbolAddress().
	 */
	CUDAErrorAddressOfConstant CUDAError = 22

	/**
	 * This indicated that a texture fetch was not able to be performed.
	 * This was previously used for device emulation of texture operations.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorTextureFetchFailed CUDAError = 23

	/**
	 * This indicated that a texture was not bound for access.
	 * This was previously used for device emulation of texture operations.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorTextureNotBound CUDAError = 24

	/**
	 * This indicated that a synchronization operation had failed.
	 * This was previously used for some device emulation functions.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorSynchronizationError CUDAError = 25

	/**
	 * This indicates that a non-float texture was being accessed with linear
	 * filtering. This is not supported by CUDA.
	 */
	CUDAErrorInvalidFilterSetting CUDAError = 26

	/**
	 * This indicates that an attempt was made to read a non-float texture as a
	 * normalized float. This is not supported by CUDA.
	 */
	CUDAErrorInvalidNormSetting CUDAError = 27

	/**
	 * Mixing of device and device emulation code was not allowed.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorMixedDeviceExecution CUDAError = 28

	/**
	 * This indicates that a CUDA Runtime API call cannot be executed because
	 * it is being called during process shut down at a point in time after
	 * CUDA driver has been unloaded.
	 */
	CUDAErrorCudartUnloading CUDAError = 29

	/**
	 * This indicates that an unknown internal error has occurred.
	 */
	CUDAErrorUnknown CUDAError = 30

	/**
	 * This indicates that the API call is not yet implemented. Production
	 * releases of CUDA will never return this error.
	 * \deprecated
	 * This error return is deprecated as of CUDA 4.1.
	 */
	CUDAErrorNotYetImplemented CUDAError = 31

	/**
	 * This indicated that an emulated device pointer exceeded the 32-bit address
	 * range.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.1. Device emulation mode was
	 * removed with the CUDA 3.1 release.
	 */
	CUDAErrorMemoryValueTooLarge CUDAError = 32

	/**
	 * This indicates that a resource handle passed to the API call was not
	 * valid. Resource handles are opaque types like ::CUDAStream_t and
	 * ::CUDAEvent_t.
	 */
	CUDAErrorInvalidResourceHandle CUDAError = 33

	/**
	 * This indicates that asynchronous operations issued previously have not
	 * completed yet. This result is not actually an error but must be indicated
	 * differently than ::CUDASuccess (which indicates completion). Calls that
	 * may return this value include ::CUDAEventQuery() and ::CUDAStreamQuery().
	 */
	CUDAErrorNotReady CUDAError = 34

	/**
	 * This indicates that the installed NVIDIA CUDA driver is older than the
	 * CUDA runtime library. This is not a supported configuration. Users should
	 * install an updated NVIDIA display driver to allow the application to run.
	 */
	CUDAErrorInsufficientDriver CUDAError = 35

	/**
	 * This indicates that the user has called ::CUDASetValidDevices()
	 * ::CUDASetDeviceFlags() ::CUDAD3D9SetDirect3DDevice()
	 * ::CUDAD3D10SetDirect3DDevice ::CUDAD3D11SetDirect3DDevice() or
	 * ::CUDAVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
	 * calling non-device management operations (allocating memory and
	 * launching kernels are examples of non-device management operations).
	 * This error can also be returned if using runtime/driver
	 * interoperability and there is an existing ::CUcontext active on the
	 * host thread.
	 */
	CUDAErrorSetOnActiveProcess CUDAError = 36

	/**
	 * This indicates that the surface passed to the API call is not a valid
	 * surface.
	 */
	CUDAErrorInvalidSurface CUDAError = 37

	/**
	 * This indicates that no CUDA-capable devices were detected by the installed
	 * CUDA driver.
	 */
	CUDAErrorNoDevice CUDAError = 38

	/**
	 * This indicates that an uncorrectable ECC error was detected during
	 * execution.
	 */
	CUDAErrorECCUncorrectable CUDAError = 39

	/**
	 * This indicates that a link to a shared object failed to resolve.
	 */
	CUDAErrorSharedObjectSymbolNotFound CUDAError = 40

	/**
	 * This indicates that initialization of a shared object failed.
	 */
	CUDAErrorSharedObjectInitFailed CUDAError = 41

	/**
	 * This indicates that the ::CUDALimit passed to the API call is not
	 * supported by the active device.
	 */
	CUDAErrorUnsupportedLimit CUDAError = 42

	/**
	 * This indicates that multiple global or constant variables (across separate
	 * CUDA source files in the application) share the same string name.
	 */
	CUDAErrorDuplicateVariableName CUDAError = 43

	/**
	 * This indicates that multiple textures (across separate CUDA source
	 * files in the application) share the same string name.
	 */
	CUDAErrorDuplicateTextureName CUDAError = 44

	/**
	 * This indicates that multiple surfaces (across separate CUDA source
	 * files in the application) share the same string name.
	 */
	CUDAErrorDuplicateSurfaceName CUDAError = 45

	/**
	 * This indicates that all CUDA devices are busy or unavailable at the current
	 * time. Devices are often busy/unavailable due to use of
	 * ::CUDAComputeModeExclusive ::CUDAComputeModeProhibited or when long
	 * running CUDA kernels have filled up the GPU and are blocking new work
	 * from starting. They can also be unavailable due to memory constraints
	 * on a device that already has active CUDA work being performed.
	 */
	CUDAErrorDevicesUnavailable CUDAError = 46

	/**
	 * This indicates that the device kernel image is invalid.
	 */
	CUDAErrorInvalidKernelImage CUDAError = 47

	/**
	 * This indicates that there is no kernel image available that is suitable
	 * for the device. This can occur when a user specifies code generation
	 * options for a particular CUDA source file that do not include the
	 * corresponding device configuration.
	 */
	CUDAErrorNoKernelImageForDevice CUDAError = 48

	/**
	 * This indicates that the current context is not compatible with this
	 * the CUDA Runtime. This can only occur if you are using CUDA
	 * Runtime/Driver interoperability and have created an existing Driver
	 * context using the driver API. The Driver context may be incompatible
	 * either because the Driver context was created using an older version
	 * of the API because the Runtime API call expects a primary driver
	 * context and the Driver context is not primary or because the Driver
	 * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions
	 * with the CUDA Driver API" for more information.
	 */
	CUDAErrorIncompatibleDriverContext CUDAError = 49

	/**
	 * This error indicates that a call to ::CUDADeviceEnablePeerAccess() is
	 * trying to re-enable peer addressing on from a context which has already
	 * had peer addressing enabled.
	 */
	CUDAErrorPeerAccessAlreadyEnabled CUDAError = 50

	/**
	 * This error indicates that ::CUDADeviceDisablePeerAccess() is trying to
	 * disable peer addressing which has not been enabled yet via
	 * ::CUDADeviceEnablePeerAccess().
	 */
	CUDAErrorPeerAccessNotEnabled CUDAError = 51

	/**
	 * This indicates that a call tried to access an exclusive-thread device that
	 * is already in use by a different thread.
	 */
	CUDAErrorDeviceAlreadyInUse CUDAError = 54

	/**
	 * This indicates profiler is not initialized for this run. This can
	 * happen when the application is running with external profiling tools
	 * like visual profiler.
	 */
	CUDAErrorProfilerDisabled CUDAError = 55

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to attempt to enable/disable the profiling via ::CUDAProfilerStart or
	 * ::CUDAProfilerStop without initialization.
	 */
	CUDAErrorProfilerNotInitialized CUDAError = 56

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call CUDAProfilerStart() when profiling is already enabled.
	 */
	CUDAErrorProfilerAlreadyStarted CUDAError = 57

	/**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call CUDAProfilerStop() when profiling is already disabled.
	 */
	CUDAErrorProfilerAlreadyStopped CUDAError = 58

	/**
	 * An assert triggered in device code during kernel execution. The device
	 * cannot be used again until ::CUDAThreadExit() is called. All existing
	 * allocations are invalid and must be reconstructed if the program is to
	 * continue using CUDA.
	 */
	CUDAErrorAssert CUDAError = 59

	/**
	 * This error indicates that the hardware resources required to enable
	 * peer access have been exhausted for one or more of the devices
	 * passed to ::CUDAEnablePeerAccess().
	 */
	CUDAErrorTooManyPeers CUDAError = 60

	/**
	 * This error indicates that the memory range passed to ::CUDAHostRegister()
	 * has already been registered.
	 */
	CUDAErrorHostMemoryAlreadyRegistered CUDAError = 61

	/**
	 * This error indicates that the pointer passed to ::CUDAHostUnregister()
	 * does not correspond to any currently registered memory region.
	 */
	CUDAErrorHostMemoryNotRegistered CUDAError = 62

	/**
	 * This error indicates that an OS call failed.
	 */
	CUDAErrorOperatingSystem CUDAError = 63

	/**
	 * This error indicates that P2P access is not supported across the given
	 * devices.
	 */
	CUDAErrorPeerAccessUnsupported CUDAError = 64

	/**
	 * This error indicates that a device runtime grid launch did not occur
	 * because the depth of the child grid would exceed the maximum supported
	 * number of nested grid launches.
	 */
	CUDAErrorLaunchMaxDepthExceeded CUDAError = 65

	/**
	 * This error indicates that a grid launch did not occur because the kernel
	 * uses file-scoped textures which are unsupported by the device runtime.
	 * Kernels launched via the device runtime only support textures created with
	 * the Texture Object API's.
	 */
	CUDAErrorLaunchFileScopedTex CUDAError = 66

	/**
	 * This error indicates that a grid launch did not occur because the kernel
	 * uses file-scoped surfaces which are unsupported by the device runtime.
	 * Kernels launched via the device runtime only support surfaces created with
	 * the Surface Object API's.
	 */
	CUDAErrorLaunchFileScopedSurf CUDAError = 67

	/**
	 * This error indicates that a call to ::CUDADeviceSynchronize made from
	 * the device runtime failed because the call was made at grid depth greater
	 * than than either the default (2 levels of grids) or user specified device
	 * limit ::CUDALimitDevRuntimeSyncDepth. To be able to synchronize on
	 * launched grids at a greater depth successfully the maximum nested
	 * depth at which ::CUDADeviceSynchronize will be called must be specified
	 * with the ::CUDALimitDevRuntimeSyncDepth limit to the ::CUDADeviceSetLimit
	 * api before the host-side launch of a kernel using the device runtime.
	 * Keep in mind that additional levels of sync depth require the runtime
	 * to reserve large amounts of device memory that cannot be used for
	 * user allocations.
	 */
	CUDAErrorSyncDepthExceeded CUDAError = 68

	/**
	 * This error indicates that a device runtime grid launch failed because
	 * the launch would exceed the limit ::CUDALimitDevRuntimePendingLaunchCount.
	 * For this launch to proceed successfully ::CUDADeviceSetLimit must be
	 * called to set the ::CUDALimitDevRuntimePendingLaunchCount to be higher
	 * than the upper bound of outstanding launches that can be issued to the
	 * device runtime. Keep in mind that raising the limit of pending device
	 * runtime launches will require the runtime to reserve device memory that
	 * cannot be used for user allocations.
	 */
	CUDAErrorLaunchPendingCountExceeded CUDAError = 69

	/**
	 * This error indicates the attempted operation is not permitted.
	 */
	CUDAErrorNotPermitted CUDAError = 70

	/**
	 * This error indicates the attempted operation is not supported
	 * on the current system or device.
	 */
	CUDAErrorNotSupported CUDAError = 71

	/**
	 * Device encountered an error in the call stack during kernel execution
	 * possibly due to stack corruption or exceeding the stack size limit.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorHardwareStackError CUDAError = 72

	/**
	 * The device encountered an illegal instruction during kernel execution
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorIllegalInstruction CUDAError = 73

	/**
	 * The device encountered a load or store instruction
	 * on a memory address which is not aligned.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorMisalignedAddress CUDAError = 74

	/**
	 * While executing a kernel the device encountered an instruction
	 * which can only operate on memory locations in certain address spaces
	 * (global shared or local) but was supplied a memory address not
	 * belonging to an allowed address space.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorInvalidAddressSpace CUDAError = 75

	/**
	 * The device encountered an invalid program counter.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorInvalidPc CUDAError = 76

	/**
	 * The device encountered a load or store instruction on an invalid memory address.
	 * This leaves the process in an inconsistent state and any further CUDA work
	 * will return the same error. To continue using CUDA the process must be terminated
	 * and relaunched.
	 */
	CUDAErrorIllegalAddress CUDAError = 77

	/**
	 * A PTX compilation failed. The runtime may fall back to compiling PTX if
	 * an application does not contain a suitable binary for the current device.
	 */
	CUDAErrorInvalidPtx CUDAError = 78

	/**
	 * This indicates an error with the OpenGL or DirectX context.
	 */
	CUDAErrorInvalidGraphicsContext CUDAError = 79

	/**
	 * This indicates that an uncorrectable NVLink error was detected during the
	 * execution.
	 */
	CUDAErrorNvlinkUncorrectable CUDAError = 80

	/**
	 * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
	 * library is used for PTX compilation. The runtime may fall back to compiling PTX
	 * if an application does not contain a suitable binary for the current device.
	 */
	CUDAErrorJitCompilerNotFound CUDAError = 81

	/**
	 * This error indicates that the number of blocks launched per grid for a kernel that was
	 * launched via either ::CUDALaunchCooperativeKernel or ::CUDALaunchCooperativeKernelMultiDevice
	 * exceeds the maximum number of blocks as allowed by ::CUDAOccupancyMaxActiveBlocksPerMultiprocessor
	 * or ::CUDAOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
	 * as specified by the device attribute ::CUDADevAttrMultiProcessorCount.
	 */
	CUDAErrorCooperativeLaunchTooLarge CUDAError = 82

	/**
	 * This indicates an internal startup failure in the CUDA runtime.
	 */
	CUDAErrorStartupFailure CUDAError = 0x7f

	/**
	 * Any unhandled CUDA driver error is added to this value and returned via
	 * the runtime. Production releases of CUDA should not return such errors.
	 * \deprecated
	 * This error return is deprecated as of CUDA 4.1.
	 */
	CUDAErrorApiFailureBase CUDAError = 10000
)
