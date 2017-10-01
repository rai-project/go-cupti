TARGETS = vector_add

all: $(TARGETS)

NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS=--cudart=shared

vector_add: vector_add.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

vector_add.so: vector_add.cu
	$(NVCC) $(NVCC_FLAGS) $^ --compiler-options '-fPIC' --shared -o $@

vector_add.ptx: vector_add.cu
	$(NVCC) --ptx $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o vec
