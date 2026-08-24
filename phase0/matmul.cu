#include<stdio.h>
#include<stdlib.h>
#include<math.h>
__global__ void matmul_kernel(float *A, float *B, float*C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    for(int i = 0; i < M * K; i++) A[i] = (float)(rand() % 10);
    for(int i = 0; i < K * N; i++) B[i] = (float)(rand() % 10);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    dim3 blocks ((N + 15)/16, (M + 15)/16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    matmul_kernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_kernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 10.0f;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C_ref[row * N + col] = sum;
        }
    }
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > 1e-3) errors++;
    }
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    printf("Average: %.3f ms | %.2f GFLOPS | %s\n", avg_ms, gflops, errors == 0 ? "PASS" : "FAIL");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Here is the error from CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return 0;
}