NVCC := nvcc
NVCCFLAG := -gencode arch=compute_60,code=sm_60

all: test_sync_blocks.ptx

test_sync_blocks.ptx: test_sync_blocks.cu test_sync_blocks.h
	$(NVCC) -rdc true -ptx $(NVCCFLAG) -o $@ $<
