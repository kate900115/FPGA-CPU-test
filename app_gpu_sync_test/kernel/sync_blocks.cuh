#ifndef __APP_GPU_SYNC_TEST_SYNC_BLOCKS_CUH
#define __APP_GPU_SYNC_TEST_SYNC_BLOCKS_CUH

// Use volatile to avoid caching values.
__device__ volatile unsigned int SYNC_BLOCKS_FLAG_[16] = {0,};
__device__          unsigned int SYNC_BLOCKS_CNT_[16] = {0,};

class SyncBlocksCtx {
public:
    __device__ SyncBlocksCtx(const unsigned int block_num,
                             const unsigned int slot_idx,
                             const unsigned int tidx) :
            flag_(&SYNC_BLOCKS_FLAG_[slot_idx]),
            cnt_(&SYNC_BLOCKS_CNT_[slot_idx]),
            slot_idx_(slot_idx),
            max_old_cnt_(block_num - 1),
            is_add_(1),
            do_sync_(tidx == 0) {}

    /* Guarantee that all previous work of all threads in cooperating blocks
     * is finished and visible to all threads in the device.
     */
    __device__ inline void sync()
    {
        __threadfence();
        // Make sure that all threads in this block have done `__threadfence()`
        // before to flip `flag_`.
        __syncthreads();
        if (do_sync_) {
            if (is_add_) {
                if (atomicAdd(cnt_, 1) == max_old_cnt_)
                    *flag_ = 1;
                while (!(*flag_)) ;
            } else {
                if (atomicSub(cnt_, 1) == 1)
                    *flag_ = 0;
                while (*flag_) ;
            }
            is_add_ ^= 1;
        }
        // We need sync here because only a single thread is checking whether
        // the flag is flipped.
        __syncthreads();
    }

    /* By flipping `flag_`, counter guarantees that all previous work of all
     * threads in this block is finished and visible to all threads in the
     * device. It does not wait for other threads outside of this block.
     */
    __device__ inline void counter()
    {
        __threadfence();
        __syncthreads();
        if (do_sync_) {
            if (is_add_) {
                if (atomicAdd(cnt_, 1) == max_old_cnt_)
                    *flag_ = 1;
            } else {
                if (atomicSub(cnt_, 1) == 1)
                    *flag_ = 0;
            }
        }
        // This is not needed for threads of which `do_sync_` is false, but
        // doing this all together slightly improves throughput in this test.
        is_add_ ^= 1;
    }

    /* Barrier guarantees that all threads in cooperating blocks have finished
     * `__threadfence()` in the previous counter.
     */
    __device__ inline void barrier()
    {
        if (do_sync_) {
            if (is_add_) {
                while (*flag_) ;
            } else {
                while (!(*flag_)) ;
            }
        }
        // We need sync here because only a single thread is checking whether
        // the flag is flipped. Alternative implementation can let each thread
        // check the flag by itself, but it is slower in this test.
        __syncthreads();
    }

private:
    volatile unsigned int *flag_;
             unsigned int *cnt_;
    const unsigned int slot_idx_;
    const unsigned int max_old_cnt_;
    int is_add_;
    int do_sync_;
};

#endif /* __APP_GPU_SYNC_TEST_SYNC_BLOCKS_CUH */
