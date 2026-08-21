#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello CUDA from thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error from CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Kernel Completed Successfully!!!\n");
    return 0;
}
