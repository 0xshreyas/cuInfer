#include <stdio.h>

__global__ void within_warp(float *out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = i;
    if (threadIdx.x % 2 == 0) { for (int k = 0; k < iters; k++) x = sinf(x) + 1.0f; }
    else                      { for (int k = 0; k < iters; k++) x = cosf(x) + 1.0f; }
    out[i] = x;
}

__global__ void between_warps(float *out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = i;
    if ((threadIdx.x / 32) % 2 == 0) { for (int k = 0; k < iters; k++) x = sinf(x) + 1.0f; }
    else                             { for (int k = 0; k < iters; k++) x = cosf(x) + 1.0f; }
    out[i] = x;
}

int main() {
    int n = 1 << 20, iters = 1000;
    float *d; cudaMalloc(&d, n * sizeof(float));
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    within_warp<<<n/256, 256>>>(d, iters);    // warmup
    between_warps<<<n/256, 256>>>(d, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    within_warp<<<n/256, 256>>>(d, iters);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("split inside each warp:  %.2f ms\n", ms);

    cudaEventRecord(s);
    between_warps<<<n/256, 256>>>(d, iters);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("split between warps:     %.2f ms\n", ms);

    cudaFree(d);
    return 0;
}