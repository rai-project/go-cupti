package cupti

var (
	DefaultEvents = []string{
		"inst_executed",                // Number of instructions executed per warp.
		"warps_launched",               // Number of warps launched.
		"l2_subp0_read_sector_misses",  // Number of read requests sent to DRAM from slice 0 of L2 cache. This increments by 1 for each 32-byte access.
		"l2_subp1_read_sector_misses",  // Number of read requests sent to DRAM from slice 1 of L2 cache. This increments by 1 for each 32-byte access.
		"l2_subp0_write_sector_misses", // "Number of write requests sent to DRAM from slice 0 of L2 cache. This increments by 1 for each 32-byte access.
		"l2_subp1_write_sector_misses", // "Number of write requests sent to DRAM from slice 1 of L2 cache. This increments by 1 for each 32-byte access.
	}
)
