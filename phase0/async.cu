#include <stdio.h>
#include <time.h>

__global__ void slow_kernel(float *out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = i;
    for (int k = 0; k < iters; k++) x = sinf(x) + 1.0f;
    out[i] = x;
}

double now_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main() {
    int n = 1 << 20;
    float *d; cudaMalloc(&d, n * sizeof(float));
    slow_kernel<<<n/256, 256>>>(d, 1000);          // warmup
    cudaDeviceSynchronize();

    // 1: CPU clock, no sync
    double t0 = now_ms();
    slow_kernel<<<n/256, 256>>>(d, 5000);
    double t1 = now_ms();
    printf("CPU clock, no sync:    %8.3f ms\n", t1 - t0);
    cudaDeviceSynchronize();

    // 2: CPU clock, with sync
    t0 = now_ms();
    slow_kernel<<<n/256, 256>>>(d, 5000);
    cudaDeviceSynchronize();
    t1 = now_ms();
    printf("CPU clock, with sync:  %8.3f ms\n", t1 - t0);

    // 3: CUDA events
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;
    cudaEventRecord(s);
    slow_kernel<<<n/256, 256>>>(d, 5000);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("CUDA events:           %8.3f ms\n", ms);

    cudaFree(d);
    return 0;
}
