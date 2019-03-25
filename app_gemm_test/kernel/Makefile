NVCC     := nvcc
NVCCFLAG := -gencode arch=compute_60,code=sm_60 -std c++14
INCLUDE  := -I../../include

all: test_gemm_kernel.ptx

%.ptx: %.cu test_gemm_kernel.h
	$(NVCC) -rdc true -ptx $(NVCCFLAG) $(INCLUDE) -o $@ $<

clean:
	rm -f test_gemm_kernel.ptx
