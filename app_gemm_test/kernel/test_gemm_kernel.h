#ifndef __APP_GEMM_TEST_TEST_GEMM_KERNEL_H
#define __APP_GEMM_TEST_TEST_GEMM_KERNEL_H

#define KERNEL_PRECISION    'h'  // 's' (single), 'h' (half)
#define KERNEL_TILE_M       32
#define KERNEL_TILE_N       32
#define KERNEL_TILE_K       32

// Perform (M x K) * (K x N) matrix multiplication. Assume all data is aligned
// in row-major.
#define M   192
#define N   768
#define K   768

#endif /* __APP_GEMM_TEST_TEST_GEMM_KERNEL_H */
