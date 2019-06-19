package cupti

var (
	DefaultEvents = []string{
		"inst_executed", // Number of instructions executed per warp.
		// "l2_subp1_read_sysmem_sector_queries", // Number of system memory read requests to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
	}
)
