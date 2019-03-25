CUDA_PATH := /usr/local/cuda-9.0
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin g++
LDFLAGS := -lcuda -lcublas
GENCODE := -gencode arch=compute_60,code=sm_60
INCLUDE := -I$(CUDA_PATH)/include -I../include

all: test_gemm

test_gemm: test_gemm.cu kernel/test_gemm_kernel.h
	$(NVCC) $(LDFLAGS) $(INCLUDE) $(GENCODE) -o test_gemm $<

clean:
	rm -f test_gemm
