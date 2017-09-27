//go:generate enumer -type=CUpti_nvtx_api_trace_cbid -json

package cupti

type CUpti_nvtx_api_trace_cbid int

const (
	CUPTI_CBID_NVTX_INVALID                          CUpti_nvtx_api_trace_cbid = 0
	CUPTI_CBID_NVTX_nvtxMarkA                        CUpti_nvtx_api_trace_cbid = 1
	CUPTI_CBID_NVTX_nvtxMarkW                        CUpti_nvtx_api_trace_cbid = 2
	CUPTI_CBID_NVTX_nvtxMarkEx                       CUpti_nvtx_api_trace_cbid = 3
	CUPTI_CBID_NVTX_nvtxRangeStartA                  CUpti_nvtx_api_trace_cbid = 4
	CUPTI_CBID_NVTX_nvtxRangeStartW                  CUpti_nvtx_api_trace_cbid = 5
	CUPTI_CBID_NVTX_nvtxRangeStartEx                 CUpti_nvtx_api_trace_cbid = 6
	CUPTI_CBID_NVTX_nvtxRangeEnd                     CUpti_nvtx_api_trace_cbid = 7
	CUPTI_CBID_NVTX_nvtxRangePushA                   CUpti_nvtx_api_trace_cbid = 8
	CUPTI_CBID_NVTX_nvtxRangePushW                   CUpti_nvtx_api_trace_cbid = 9
	CUPTI_CBID_NVTX_nvtxRangePushEx                  CUpti_nvtx_api_trace_cbid = 10
	CUPTI_CBID_NVTX_nvtxRangePop                     CUpti_nvtx_api_trace_cbid = 11
	CUPTI_CBID_NVTX_nvtxNameCategoryA                CUpti_nvtx_api_trace_cbid = 12
	CUPTI_CBID_NVTX_nvtxNameCategoryW                CUpti_nvtx_api_trace_cbid = 13
	CUPTI_CBID_NVTX_nvtxNameOsThreadA                CUpti_nvtx_api_trace_cbid = 14
	CUPTI_CBID_NVTX_nvtxNameOsThreadW                CUpti_nvtx_api_trace_cbid = 15
	CUPTI_CBID_NVTX_nvtxNameCuDeviceA                CUpti_nvtx_api_trace_cbid = 16
	CUPTI_CBID_NVTX_nvtxNameCuDeviceW                CUpti_nvtx_api_trace_cbid = 17
	CUPTI_CBID_NVTX_nvtxNameCuContextA               CUpti_nvtx_api_trace_cbid = 18
	CUPTI_CBID_NVTX_nvtxNameCuContextW               CUpti_nvtx_api_trace_cbid = 19
	CUPTI_CBID_NVTX_nvtxNameCuStreamA                CUpti_nvtx_api_trace_cbid = 20
	CUPTI_CBID_NVTX_nvtxNameCuStreamW                CUpti_nvtx_api_trace_cbid = 21
	CUPTI_CBID_NVTX_nvtxNameCuEventA                 CUpti_nvtx_api_trace_cbid = 22
	CUPTI_CBID_NVTX_nvtxNameCuEventW                 CUpti_nvtx_api_trace_cbid = 23
	CUPTI_CBID_NVTX_nvtxNameCudaDeviceA              CUpti_nvtx_api_trace_cbid = 24
	CUPTI_CBID_NVTX_nvtxNameCudaDeviceW              CUpti_nvtx_api_trace_cbid = 25
	CUPTI_CBID_NVTX_nvtxNameCudaStreamA              CUpti_nvtx_api_trace_cbid = 26
	CUPTI_CBID_NVTX_nvtxNameCudaStreamW              CUpti_nvtx_api_trace_cbid = 27
	CUPTI_CBID_NVTX_nvtxNameCudaEventA               CUpti_nvtx_api_trace_cbid = 28
	CUPTI_CBID_NVTX_nvtxNameCudaEventW               CUpti_nvtx_api_trace_cbid = 29
	CUPTI_CBID_NVTX_nvtxDomainMarkEx                 CUpti_nvtx_api_trace_cbid = 30
	CUPTI_CBID_NVTX_nvtxDomainRangeStartEx           CUpti_nvtx_api_trace_cbid = 31
	CUPTI_CBID_NVTX_nvtxDomainRangeEnd               CUpti_nvtx_api_trace_cbid = 32
	CUPTI_CBID_NVTX_nvtxDomainRangePushEx            CUpti_nvtx_api_trace_cbid = 33
	CUPTI_CBID_NVTX_nvtxDomainRangePop               CUpti_nvtx_api_trace_cbid = 34
	CUPTI_CBID_NVTX_nvtxDomainResourceCreate         CUpti_nvtx_api_trace_cbid = 35
	CUPTI_CBID_NVTX_nvtxDomainResourceDestroy        CUpti_nvtx_api_trace_cbid = 36
	CUPTI_CBID_NVTX_nvtxDomainNameCategoryA          CUpti_nvtx_api_trace_cbid = 37
	CUPTI_CBID_NVTX_nvtxDomainNameCategoryW          CUpti_nvtx_api_trace_cbid = 38
	CUPTI_CBID_NVTX_nvtxDomainRegisterStringA        CUpti_nvtx_api_trace_cbid = 39
	CUPTI_CBID_NVTX_nvtxDomainRegisterStringW        CUpti_nvtx_api_trace_cbid = 40
	CUPTI_CBID_NVTX_nvtxDomainCreateA                CUpti_nvtx_api_trace_cbid = 41
	CUPTI_CBID_NVTX_nvtxDomainCreateW                CUpti_nvtx_api_trace_cbid = 42
	CUPTI_CBID_NVTX_nvtxDomainDestroy                CUpti_nvtx_api_trace_cbid = 43
	CUPTI_CBID_NVTX_nvtxDomainSyncUserCreate         CUpti_nvtx_api_trace_cbid = 44
	CUPTI_CBID_NVTX_nvtxDomainSyncUserDestroy        CUpti_nvtx_api_trace_cbid = 45
	CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireStart   CUpti_nvtx_api_trace_cbid = 46
	CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireFailed  CUpti_nvtx_api_trace_cbid = 47
	CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireSuccess CUpti_nvtx_api_trace_cbid = 48
	CUPTI_CBID_NVTX_nvtxDomainSyncUserReleasing      CUpti_nvtx_api_trace_cbid = 49
	CUPTI_CBID_NVTX_SIZE
	CUPTI_CBID_NVTX_FORCE_INT CUpti_nvtx_api_trace_cbid = 0x7fffffff
)
