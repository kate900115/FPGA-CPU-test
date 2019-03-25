NVCC := nvcc -ccbin g++
LDFLAGS := -lcuda
INCLUDE := -I/usr/local/cuda/include

all: test_sync_blocks

test_sync_blocks: test_sync_blocks.c kernel/test_sync_blocks.h
	$(NVCC) $(LDFLAGS) $(INCLUDE) -o $@ $<

clean:
	rm -f test_sync_blocks
