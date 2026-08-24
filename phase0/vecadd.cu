#include<stdio.h>

__global__ void vecadd_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1000000;
    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *c = (float*)malloc(N * sizeof(float));
    float *d_a, *d_b, *d_c;
    int errors = 0;
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) errors++;
    }
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecadd_kernel<<<blocks, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error from CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (errors == 0) printf("PASS\n");
    else printf("FAIL: %d errors\n", errors);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}