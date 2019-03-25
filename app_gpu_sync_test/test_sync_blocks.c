#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "kernel/test_sync_blocks.h"

#define EXIT_SUCCESS    0
#define EXIT_ERROR      1
#define BLOCK_SIZE      32

#define CG_LIB_FILE     "/usr/local/cuda/lib64/libcudadevrt.a"
#define PTX_FILE        "kernel/test_sync_blocks.ptx"
#define KERNEL_NAME     "test_sync_blocks_kernel"

static void checkError(CUresult status)
{
    if (status == CUDA_SUCCESS)
        return;
    const char *perr;
    CUresult ok = cuGetErrorString(status, &perr);
    if (ok == CUDA_SUCCESS) {
        if (perr) {
            fprintf(stderr, "error(%d) info: %s\n", status, perr);
        } else {
            fprintf(stderr, "error(%d) info: unknown\n", status);
        }
    }
    exit(EXIT_ERROR);
}

static double timerSec(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        fprintf(stderr, "timer error: clock_gettime failed.\n");
        exit(EXIT_ERROR);
    }
	return (tspec.tv_nsec / 1000000000.0 + tspec.tv_sec);
}

/* Test of grid synchronization using CUDA driver APIs. */
int main()
{
    CUdevice device;
    CUcontext context;
    CUlinkState lstate;
    CUmodule module;
    CUfunction function;
    int pi;

    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));

    // Verify if CG feature is supported
    checkError(cuDeviceGetAttribute(
        &pi, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device));
    if (pi != 1) {
        fprintf(stderr, "This GPU does not support cooperative launch. Exit\n");
        return EXIT_ERROR;
    }

    // Get the number of total MP
    checkError(cuDeviceGetAttribute(&pi,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    const int mpNum = pi;
    printf("# Multiprocessor: %d\n", mpNum);

    checkError(cuCtxCreate(&context, 0, device));

    // JIT compilation to link CG library to the test kernel
    void *cubin;
    size_t cubin_size;
    checkError(cuLinkCreate(0, 0, 0, &lstate));
    checkError(cuLinkAddFile(lstate, CU_JIT_INPUT_PTX, PTX_FILE, 0, 0, 0));
    checkError(cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, CG_LIB_FILE, 0, 0, 0));
    checkError(cuLinkComplete(lstate, &cubin, &cubin_size));
    printf("JIT compilation done.\n");

#ifdef USE_ATOMIC_SYNC
#ifdef USE_STAGED_SYNC
    printf("Synchronization test: atomic-based (staged), %d blocks\n", NUM_BLOCK_TO_SYNC);
#else
    printf("Synchronization test: atomic-based, %d blocks\n", NUM_BLOCK_TO_SYNC);
#endif  // USE_STAGED_SYNC
#else
    printf("Synchronization test: cooperative groups, %d blocks\n",
           NUM_BLOCK_TO_SYNC);
#endif

    // Load cubin and get the kernel function pointer
    checkError(cuModuleLoadData(&module, cubin));
    checkError(cuModuleGetFunction(&function, module, KERNEL_NAME));
    checkError(cuLinkDestroy(lstate));

    // Prepare a device array for sync test
    CUdeviceptr pDev;
    int *pHost;
    int length = NUM_BLOCK_TO_SYNC * BLOCK_SIZE;
    checkError(cuMemAllocHost((void **)&pHost, length * sizeof(int)));
    checkError(cuMemAlloc(&pDev, length * sizeof(int)));
    checkError(cuMemcpyHtoD(pDev, (const void *)pHost, length * sizeof(int)));

    checkError(cuCtxSynchronize());
    fflush(stdout);

    // Kernel launch
    void *args[2] = {&pDev, &length};
    double time_start = timerSec();
    checkError(cuLaunchCooperativeKernel(function, NUM_BLOCK_TO_SYNC, 1, 1,
                                         BLOCK_SIZE, 1, 1, 0, 0, args));
    checkError(cuCtxSynchronize());
    double time_elapsed = timerSec() - time_start;
    printf("Elapsed %.2f sec\n", time_elapsed);

    // Check correctness and exit
    checkError(cuMemcpyDtoH((void *)pHost, pDev, length * sizeof(int)));
    int i;
    int ret = EXIT_SUCCESS;
    for (i = 0; i < length; ++i) {
        if (pHost[i] != 400001) {
#ifndef USE_STAGED_SYNC
            fprintf(stderr, "Synchronization test failed. "
                    "Expected: 400001, actual: %d.\n", pHost[i]);
            ret = EXIT_ERROR;
            break;
#endif  // USE_STAGED_SYNC
        }
        if (i == length - 1) {
            printf("Synchronization test succeed.\n");
        }
    }
    cuMemFreeHost(pHost);
    cuMemFree(pDev);
    cuCtxDestroy(context);
    return ret;
}
