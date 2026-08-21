#include<stdio.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device:            %s\n", p.name);
    printf("Compute Capability:%d.%d\n", p.major, p.minor);
    printf("SMs:               %d\n", p.multiProcessorCount);
    printf("Shared mem/Block:  %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("Max Threads/Block: %d\n", p.maxThreadsPerBlock);
    printf("Warp Size:         %d\n", p.warpSize);
    printf("Global Memory:     %.1f GB\n", p.totalGlobalMem / 1e9);
    return 0;
}
