#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../gemm/include/openai_gemm_utils.h"

#include "kernel/test_gemm_kernel.h"
#include "fp16_conversion.h"

#define EXIT_SUCCESS    0
#define EXIT_ERROR      1

#define CG_LIB_FILE     "/usr/local/cuda/lib64/libcudadevrt.a"
#define GEMM_CUBIN_FILE "../gemm/openai-gemm.cubin"
#define TEST_PTX_FILE   "kernel/test_gemm_kernel.ptx"
#define KERNEL_NAME     "test_gemm_kernel"

#define TOTAL_GEMM_NUM          1000000
#define PER_KERNEL_GEMM_NUM     100
#define DO_CORRECT_TEST         1

static void checkError(int status)
{
    if (status == CUDA_SUCCESS)
        return;
    const char *perr;
    CUresult ok = cuGetErrorString((CUresult)status, &perr);
    if (ok == CUDA_SUCCESS) {
        if (perr) {
            fprintf(stderr, "Error(%d) info: %s\n", status, perr);
        } else {
            fprintf(stderr, "Error(%d) info: unknown\n", status);
        }
    }
    exit(EXIT_ERROR);
}

static long timerNsec(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        fprintf(stderr, "Error: clock_gettime failed.\n");
        return -1;
    }
	return (tspec.tv_sec * (1000000000L) + tspec.tv_nsec);
}

static float randFloat(float maxVal)
{
    int mid = RAND_MAX / 2;
    return (float)(rand() - mid) / (float)mid * maxVal;
}

static void reportAccuracy(float *cpuC, void *gpuC, int lenC, char precision)
{
    float diff, res, cpuVal = 0, gpuVal = 0, cpuSum = 0, maxErr = 0;
    for (int i = 0; i < lenC; ++i) {
        if (precision == 's') {
            res = ((float *)gpuC)[i];
        } else {
            res = half_to_float(((__half *)gpuC)[i]);
        }
        cpuSum += cpuC[i] > 0 ? cpuC[i] : -cpuC[i];
        diff = cpuC[i] - res;
        if (diff > maxErr) {
            maxErr = diff;
            cpuVal = cpuC[i];
            gpuVal = res;
        } else if (diff < -maxErr) {
            maxErr = -diff;
            cpuVal = cpuC[i];
            gpuVal = res;
        }
    }
    float pctErr = maxErr * lenC * 100. / cpuSum;
    float pctErrLimit = (precision == 's') ? 0.005 : 0.7;
    if (pctErr > pctErrLimit) {
        printf("[FAILED] error rate: %.3f%% (expected: < %.3f%%). "
               "GPU result: %f / CPU result: %f\n",
               pctErr, pctErrLimit, gpuVal, cpuVal);
    } else {
        printf("[PASSED] error rate: %.3f%% (expected: < %.3f%%).\n",
               pctErr, pctErrLimit);
    }
}

/* OpenAI GEMM test */
int test_openai(float *pA, float *pB, float *pC, float *pRes,
                float alpha, float beta,
                int m, int n, int k, int tileM, int tileN, int tileK,
                char precision)
{
    printf("========== Running OpenAI GEMM Test: Tile (%dx%dx%d)\n",
           tileM, tileN, tileK);
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

    checkError(cuCtxCreate(&context, 0, device));

    // JIT compilation to link needed files to the test kernel
    void *cubin;
    size_t cubin_size;
    checkError(cuLinkCreate(0, 0, 0, &lstate));
    checkError(cuLinkAddFile(lstate, CU_JIT_INPUT_PTX, TEST_PTX_FILE, 0, 0, 0));
    checkError(cuLinkAddFile(lstate, CU_JIT_INPUT_CUBIN, GEMM_CUBIN_FILE, 0, 0, 0));
    checkError(cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, CG_LIB_FILE, 0, 0, 0));
    checkError(cuLinkComplete(lstate, &cubin, &cubin_size));
    printf("JIT compilation done.\n");

    // Get the number of total MP
    checkError(cuDeviceGetAttribute(&pi,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    const int mpNum = pi;
    printf("# Multiprocessor: %d\n", mpNum);

    int *dims = gemmDim(m, n, tileM, tileN);
    printf("Requested thread block dimensions: (%d,%d,%d), size: %d\n",
           dims[0], dims[1], dims[2], dims[3]);

    // Load cubin and get the kernel function pointer
    checkError(cuModuleLoadData(&module, cubin));
    checkError(cuModuleGetFunction(&function, module, KERNEL_NAME));
    checkError(cuLinkDestroy(lstate));

    // Prepare arrays for matrices
    CUdeviceptr pDevA;
    CUdeviceptr pDevB;
    CUdeviceptr pDevC;
    void *pHostA;
    void *pHostB;
    void *pHostC;

    int lenA = m * k;
    int lenB = n * k;
    int lenC = m * n;
    int typeSize = (precision == 's') ? sizeof(float) : sizeof(__half);

    checkError(cuMemAlloc(&pDevA, lenA * typeSize));
    checkError(cuMemAlloc(&pDevB, lenB * typeSize));
    checkError(cuMemAlloc(&pDevC, lenC * typeSize));
    checkError(cuMemAllocHost(&pHostA, lenA * typeSize));
    checkError(cuMemAllocHost(&pHostB, lenB * typeSize));
    checkError(cuMemAllocHost(&pHostC, lenC * typeSize));

    int i;
    if (precision == 's') {
        for (i = 0; i < lenA; ++i)
            ((float *)pHostA)[i] = pA[i];
        for (i = 0; i < lenB; ++i)
            ((float *)pHostB)[i] = pB[i];
        for (i = 0; i < lenC; ++i)
            ((float *)pHostC)[i] = pC[i];
    } else {
        for (i = 0; i < lenA; ++i)
            ((__half *)pHostA)[i] = approx_float_to_half(pA[i]);
        for (i = 0; i < lenB; ++i)
            ((__half *)pHostB)[i] = approx_float_to_half(pB[i]);
        for (i = 0; i < lenC; ++i)
            ((__half *)pHostC)[i] = approx_float_to_half(pC[i]);
    }

    // Copy matrices to device arrays
    checkError(cuMemcpyHtoD(pDevA, pHostA, lenA * typeSize));
    checkError(cuMemcpyHtoD(pDevB, pHostB, lenB * typeSize));
    checkError(cuMemcpyHtoD(pDevC, pHostC, lenC * typeSize));

    int repeat = 1;
    void *args[] = {&pDevC, &pDevA, &pDevB, &alpha, &beta, &repeat};

    // Kernel launch
    checkError(cuLaunchCooperativeKernel(function, 
                                         dims[0], dims[1], dims[2],
                                         dims[3], 1, 1, 20000, 0, args));
    checkError(cuCtxSynchronize());

    // Correctness test
    if (DO_CORRECT_TEST) {
        checkError(cuMemcpyDtoH(pHostC, pDevC, lenC * typeSize));
        reportAccuracy(pRes, pHostC, lenC, precision);
    }
    if (TOTAL_GEMM_NUM > 0) {
        printf("Running performance test.. ");
        fflush(stdout);
        repeat = PER_KERNEL_GEMM_NUM;

        long start, end;
        start = timerNsec();
        for (i = 0; i < TOTAL_GEMM_NUM / PER_KERNEL_GEMM_NUM; ++i) {
            checkError(cuLaunchCooperativeKernel(function, 
                    dims[0], dims[1], dims[2], dims[3], 1, 1, 20000, 0, args));
        }
        checkError(cuCtxSynchronize());
        end = timerNsec();
        if (start < 0 || end < 0)
            return EXIT_ERROR;
        printf("%.2f usec/launch\n", (end - start) * 0.001 / (float)TOTAL_GEMM_NUM);
    }

    cuMemFree(pDevA);
    cuMemFree(pDevB);
    cuMemFree(pDevC);
    cuMemFreeHost(pHostA);
    cuMemFreeHost(pHostB);
    cuMemFreeHost(pHostC);
    cuCtxDestroy(context);
    return EXIT_SUCCESS;
}

/* CUBLAS GEMM test */
int test_cublas(float *pA, float *pB, float *pC, float *pRes,
                float alpha, float beta, int m, int n, int k, char precision)
{
    printf("========== Running CUBLAS GEMM Test\n");

    void *pCublasDevA;
    void *pCublasDevB;
    void *pCublasDevC;
    void *pCublasHostA;
    void *pCublasHostB;
    void *pCublasHostC;
    int lenA = m * k;
    int lenB = n * k;
    int lenC = m * n;
    int typeSize = (precision == 's') ? sizeof(float) : sizeof(__half);

    checkError(cudaMalloc(&pCublasDevA, lenA * typeSize));
    checkError(cudaMalloc(&pCublasDevB, lenB * typeSize));
    checkError(cudaMalloc(&pCublasDevC, lenC * typeSize));
    checkError(cudaMallocHost(&pCublasHostA, lenA * typeSize));
    checkError(cudaMallocHost(&pCublasHostB, lenB * typeSize));
    checkError(cudaMallocHost(&pCublasHostC, lenC * typeSize));

    int i;
    if (precision == 's') {
        for (i = 0; i < lenA; ++i)
            ((float *)pCublasHostA)[i] = pA[i];
        for (i = 0; i < lenB; ++i)
            ((float *)pCublasHostB)[i] = pB[i];
        for (i = 0; i < lenC; ++i)
            ((float *)pCublasHostC)[i] = pC[i];
    } else {
        for (i = 0; i < lenA; ++i)
            ((__half *)pCublasHostA)[i] = approx_float_to_half(pA[i]);
        for (i = 0; i < lenB; ++i)
            ((__half *)pCublasHostB)[i] = approx_float_to_half(pB[i]);
        for (i = 0; i < lenC; ++i)
            ((__half *)pCublasHostC)[i] = approx_float_to_half(pC[i]);
    }

    checkError(cudaMemcpy(pCublasDevA, pCublasHostA, lenA * typeSize, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(pCublasDevB, pCublasHostB, lenB * typeSize, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(pCublasDevC, pCublasHostC, lenC * typeSize, cudaMemcpyHostToDevice));

    __half alpha16 = approx_float_to_half(alpha);
    __half beta16 = approx_float_to_half(beta);

    cublasHandle_t handle;
    checkError(cublasCreate(&handle));

    // Kernel launch
    if (precision == 's') {
        checkError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha, (const float *)pCublasDevB, n, (const float *)pCublasDevA, k,
            &beta, (float *)pCublasDevC, n));
    } else {
        checkError(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha16, (const __half *)pCublasDevB, n, (const __half *)pCublasDevA, k,
            &beta16, (__half *)pCublasDevC, n));
    }
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    if (DO_CORRECT_TEST) {
        checkError(cudaMemcpy(pCublasHostC, pCublasDevC, lenC * typeSize, cudaMemcpyDeviceToHost));
        reportAccuracy(pRes, pCublasHostC, lenC, precision);
    }
    if (TOTAL_GEMM_NUM > 0) {
        printf("Running performance test.. ");
        fflush(stdout);

        // Measure throughput
        int i;
        long start, end;
        start = timerNsec();
        if (KERNEL_PRECISION == 's') {
            for (i = 0; i < TOTAL_GEMM_NUM; ++i) {
                checkError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    &alpha, (const float *)pCublasDevB, n, (const float *)pCublasDevA, k,
                    &beta, (float *)pCublasDevC, n));
            }
        } else {
            for (i = 0; i < TOTAL_GEMM_NUM; ++i) {
                checkError(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    &alpha16, (const __half *)pCublasDevB, n, (const __half *)pCublasDevA, k,
                    &beta16, (__half *)pCublasDevC, n));
            }
        }
        checkError(cudaDeviceSynchronize());
        end = timerNsec();
        if (start < 0 || end < 0)
            return EXIT_ERROR;
        printf("%.2f usec/launch\n", (end - start) * 0.001 / (float)TOTAL_GEMM_NUM);
    }

    cublasDestroy(handle);
    cudaFreeHost(pCublasHostA);
    cudaFreeHost(pCublasHostB);
    cudaFreeHost(pCublasHostC);
    cudaFree(pCublasDevA);
    cudaFree(pCublasDevB);
    cudaFree(pCublasDevC);
    return EXIT_SUCCESS;
}

/* Test GEMM implementations. */
int main()
{
    printf("GEMM Test Input: Precision '%c', (M,N,K)=(%d,%d,%d)\n",
        KERNEL_PRECISION, M, N, K);
    srand(time(0));

    int lenA = M * K;
    int lenB = N * K;
    int lenC = M * N;

    float *pA = (float *)malloc(lenA * sizeof(float));
    assert(pA != 0);
    float *pB = (float *)malloc(lenB * sizeof(float));
    assert(pB != 0);
    float *pC = (float *)malloc(lenC * sizeof(float));
    assert(pC != 0);
    float *pRes = (float *)malloc(lenC * sizeof(float));
    assert(pRes != 0);

    float alpha = 1.0;
    float beta = 1.0;

    int i, j, h, w;
    for (i = 0; i < lenA; ++i)
        pA[i] = randFloat(2);
    for (i = 0; i < lenB; ++i)
        pB[i] = randFloat(2);
    for (i = 0; i < lenC; ++i)
        pC[i] = randFloat(2);

    // Calculate CPU results
    float temp;
    for (i = 0; i < lenC; ++i) {
        temp = 0;
        h = i / N;
        w = i % N;
        for (j = 0; j < K; ++j) {
            temp += pA[h * K + j] * pB[j * N + w];
        }
        pRes[i] = alpha * temp + beta * pC[i];
    }

    int ret;
    ret = test_openai(pA, pB, pC, pRes, alpha, beta, M, N, K,
                      KERNEL_TILE_M, KERNEL_TILE_N, KERNEL_TILE_K,
                      KERNEL_PRECISION);
    ret |= test_cublas(pA, pB, pC, pRes, alpha, beta, M, N, K,
                       KERNEL_PRECISION);
    free(pA);
    free(pB);
    free(pC);
    return ret;
}
