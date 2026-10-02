#include <stdio.h>
#include <stdlib.h>

int main() {
    size_t bytes = 256 * 1024 * 1024;     // 256 MB
    float *h = (float*)malloc(bytes);
    float *d1, *d2;
    cudaMalloc(&d1, bytes);
    cudaMalloc(&d2, bytes);
    for (size_t i = 0; i < bytes / 4; i++) h[i] = 1.0f;

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    cudaMemcpy(d1, h, bytes, cudaMemcpyHostToDevice);   // warmup

    cudaEventRecord(s);
    cudaMemcpy(d1, h, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("RAM  -> VRAM (across PCIe):  %6.2f ms  = %6.1f GB/s\n", ms, bytes / ms / 1e6);

    cudaEventRecord(s);
    cudaMemcpy(d2, d1, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("VRAM -> VRAM (stays on GPU): %6.2f ms  = %6.1f GB/s\n", ms, 2 * bytes / ms / 1e6);

    free(h); cudaFree(d1); cudaFree(d2);
    return 0;
}
