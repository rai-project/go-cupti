TARGETS = vector_add

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared

vector_add: vector_add.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o vec
