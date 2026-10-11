NVCC  = nvcc
FLAGS = -O3 -arch=native

all: src/model bench/bench_matmul

src/model: src/model.cu src/matmul_fast.cuh src/bench_e2e.cuh
	$(NVCC) $(FLAGS) src/model.cu -o src/model

bench/bench_matmul: bench/bench_matmul.cu
	$(NVCC) $(FLAGS) bench/bench_matmul.cu -o bench/bench_matmul -lcublas

clean:
	rm -f src/model bench/bench_matmul

.PHONY: all clean
