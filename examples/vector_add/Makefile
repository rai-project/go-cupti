TARGETS = vector_add vector_add.so

all: $(TARGETS)

NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS=--cudart=shared -g -O0
LIB=-lnvToolsExt

vector_add: vector_add.cu main.cpp
	$(NVCC) $(LIB) $^ -o $@

vector_add.so: vector_add.cu
	$(NVCC) $(LIB) $(NVCC_FLAGS) $^ --compiler-options '-fPIC' --shared -o $@

vector_add.ptx: vector_add.cu
	$(NVCC) --ptx $(LIB) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o *.so vector_add
