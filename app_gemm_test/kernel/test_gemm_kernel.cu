#include <stdio.h>
#include "../../gemm/include/openai_gemm.cuh"
#include "test_gemm_kernel.h"

extern "C" {

__global__ void test_gemm_kernel(void *C, void *A, void *B,
                                 float alpha, float beta, int repeat)
{
    gemm<KERNEL_PRECISION, KERNEL_TILE_M, KERNEL_TILE_N, KERNEL_TILE_K,
         0, 0, M, N, K, K, N, N>(C, A, B, alpha, beta);
    for (int i = 0; i < repeat - 1; ++i) {
        gemm<KERNEL_PRECISION, KERNEL_TILE_M, KERNEL_TILE_N, KERNEL_TILE_K,
             0, 0, M, N, K, K, N, N>(C, A, B, alpha, beta);
    }
}

}
