#include <stdio.h>

__global__ void neighbors(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void scattered(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t j = ((size_t)i * 7919) % n;   // each thread jumps ~7919 floats from its neighbor
        out[i] = in[j];
    }
}

int main() {
    int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);
    float *in, *out;
    cudaMalloc(&in, bytes); cudaMalloc(&out, bytes);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    neighbors<<<n/256, 256>>>(in, out, n);
    scattered<<<n/256, 256>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    neighbors<<<n/256, 256>>>(in, out, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("neighboring reads: %6.3f ms  = %6.1f GB/s\n", ms, 2.0 * bytes / ms / 1e6);

    cudaEventRecord(s);
    scattered<<<n/256, 256>>>(in, out, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("scattered reads:   %6.3f ms  = %6.1f GB/s\n", ms, 2.0 * bytes / ms / 1e6);

    cudaFree(in); cudaFree(out);
    return 0;
}
