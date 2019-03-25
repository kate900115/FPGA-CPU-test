#ifndef __APP_GPU_SYNC_TEST_TEST_SYNC_BLOCKS_CUH
#define __APP_GPU_SYNC_TEST_TEST_SYNC_BLOCKS_CUH

// Use atomic-based sync by default. Comment this out to use this_grid().sync().
#define USE_ATOMIC_SYNC

#ifdef USE_ATOMIC_SYNC
// Use staged sync if defined. Enabled only when USE_ATOMIC_SYNC is defined.
// Note that this is only for throughput test of staged sync, so enabling this
// will skip the correctness check.
//#define USE_STAGED_SYNC
#endif  // USE_ATOMIC_SYNC

// Atomic-based sync needs a pre-defined number of blocks to sync.
#define NUM_BLOCK_TO_SYNC   56

#endif /* __APP_GPU_SYNC_TEST_TEST_SYNC_BLOCKS_CUH */
