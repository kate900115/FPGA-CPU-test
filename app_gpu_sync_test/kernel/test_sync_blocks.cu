#include "sync_blocks.cuh"
#include "test_sync_blocks.h"

#ifndef USE_ATOMIC_SYNC
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

extern "C" __global__ void test_sync_blocks_kernel(int *array, int length)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length)
        return;
#ifdef USE_ATOMIC_SYNC
    SyncBlocksCtx ctx = SyncBlocksCtx(NUM_BLOCK_TO_SYNC, 0, threadIdx.x);
#endif
    if (idx == 0) {
        // Thread 0 of block 0 initialize the array.
        for (int i = 0; i < length; ++i) {
            array[i] = 1;
        }
    }

    for (int i = 0; i < 200000; ++i) {
        // Wait until thread 0 of block 0 finishes work.
#ifndef USE_ATOMIC_SYNC
        this_grid().sync();
#else
#ifndef USE_STAGED_SYNC
        ctx.sync();
#else
        ctx.barrier();
        ctx.counter();
#endif // USE_STAGED_SYNC
#endif // USE_ATOMIC_SYNC

        // Increase all values by one.
        array[idx] += 1;

#ifndef USE_ATOMIC_SYNC
        this_grid().sync();
#else
#ifndef USE_STAGED_SYNC
        ctx.sync();
#else
        ctx.barrier();
        ctx.counter();
#endif // USE_STAGED_SYNC
#endif // USE_ATOMIC_SYNC

        if (idx == 0) {
            // Thread 0 of block 0 increase all values by one.
            for (int i = 0; i < length; ++i) {
                array[i] += 1;
            }
        }
    }
    // Expected result: all values are 400001.
}
