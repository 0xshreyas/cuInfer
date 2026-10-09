// bench_matmul.cu: time matmul kernels against cuBLAS on GPT-2's real shapes.
// Build: nvcc -O3 -arch=native -o bench_matmul bench_matmul.cu -lcublas
// Run:   ./bench_matmul 128        (the number is M = tokens, T)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 32

#define BM 64
#define BN 64
#define BK 8
#define TM 8



#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {          \
    printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(e),             \
           __FILE__, __LINE__); exit(1); } } while (0)

typedef void (*Launcher)(const float *A, const float *B, const float *bias,
                         float *C, int M, int N, int K);

// ---------- kernel 0: the naive kernel, same as model.cu ----------
__global__ void matmul_naive(const float *A, const float *B, const float *bias,
                             float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        if (bias) sum += bias[col];
        C[row * N + col] = sum;
    }
}
void launch_naive(const float *A, const float *B, const float *bias,
                  float *C, int M, int N, int K) {
    dim3 tpb(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<grid, tpb>>>(A, B, bias, C, M, N, K);
}

// ---------- the reference: cuBLAS ----------
cublasHandle_t g_handle;
void launch_cublas(const float *A, const float *B, const float *bias,
                   float *C, int M, int N, int K) {
    // cuBLAS assumes column-major. Asking it for C^T = B^T A^T gives
    // exactly our row-major C = A B, with no copies. (bias is ignored.)
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &one, B, N, A, K, &zero, C, N);
}

__global__ void matmul_tiled(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && t * TILE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        if (col < N && t * TILE + ty < K) {
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads(); 
    }
    if (row < M && col < N) {
        if (bias) sum += bias[col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_1d(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    int threadCol = tid % BN;
    int threadRow = tid / BN;

    int aRow = tid / BK, aCol = tid % BK;
    int bRow = tid / BN, bCol = tid % BN;

    float sum[TM] = {0.0f};
    for(int t = 0; t < (K + BK - 1) / BK; t++) {
        int aR = blockRow + aRow, aC = t * BK + aCol;
        if (aR < M && aC < K) {
            As[aRow][aCol] = A[aR * K + aC];
        } else {
            As[aRow][aCol] = 0.0f;
        }
        int bR = t * BK + bRow, bC = blockCol + bCol;
        if (bR < K && bC < N) {
            Bs[bRow][bCol] = B[bR * N + bC]; 
        } else {
            Bs[bRow][bCol] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            float b = Bs[k][threadCol];
            for (int i = 0; i < TM; i++) {
                sum[i] += As[threadRow * TM + i][k] * b;
            }
        }
        __syncthreads();
        for (int i = 0; i < TM; i++) {
            int row = blockRow + threadRow * TM + i;
            int col = blockCol + threadCol;
            if (row < M && col < N) {
                if (bias) sum[i] += bias[col];
                C[row * N + col] = sum[i];
            }
        }
    }
}

void launch_1d(const float *A, const float *B, const float *bias,
               float *C, int M, int N, int K) {
    dim3 tpb(BM * BN / TM);                         // 512 threads, numbered in a line
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_1d<<<grid, tpb>>>(A, B, bias, C, M, N, K);
}

void launch_tiled(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    dim3 tpb(TILE, TILE);
    dim3 grid((N + TILE - 1)/ TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, tpb>>>(A, B, bias, C, M, N, K);
}

struct Kernel { const char *name; Launcher fn; };
Kernel kernels[] = {
    {"cublas", launch_cublas},
    {"naive",  launch_naive},
    {"tiled",  launch_tiled},
    {"1d",     launch_1d},
    // your new kernels get added here
};
const int NK = sizeof(kernels) / sizeof(kernels[0]);

// GPT-2's four linear layers plus logits: B is (K, N)
struct Shape { const char *name; int N, K; };
Shape shapes[] = {
    {"qkv",     2304,  768},
    {"proj",     768,  768},
    {"fc",      3072,  768},
    {"fcproj",   768, 3072},
    {"logits", 50257,  768},
};
const int NS = sizeof(shapes) / sizeof(shapes[0]);

float time_ms(Launcher fn, const float *A, const float *B, float *C,
              int M, int N, int K) {
    for (int i = 0; i < 3; i++) fn(A, B, nullptr, C, M, N, K);   // warm-up
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    const int reps = 20;
    CK(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) fn(A, B, nullptr, C, M, N, K);
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));
    CK(cudaGetLastError());
    float ms;
    CK(cudaEventElapsedTime(&ms, start, stop));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    return ms / reps;
}

// Largest difference from cuBLAS, as a fraction of the largest value.
float rel_err(const float *ref, const float *out, size_t n) {
    float diff = 0.0f, scale = 0.0f;
    for (size_t i = 0; i < n; i++) {
        diff  = fmaxf(diff, fabsf(ref[i] - out[i]));
        scale = fmaxf(scale, fabsf(ref[i]));
    }
    return diff / scale;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, %d SMs, %d max threads per SM.  M = %d tokens\n\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, M);
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed\n");
        return 1;
    }

    // Buffers sized for the largest shape; smaller shapes use the front part.
    size_t maxA = (size_t)M * 3072, maxB = (size_t)768 * 50257;
    size_t maxC = (size_t)M * 50257;
    float *hA   = (float *)malloc(maxA * sizeof(float));
    float *hB   = (float *)malloc(maxB * sizeof(float));
    float *hC   = (float *)malloc(maxC * sizeof(float));
    float *hRef = (float *)malloc(maxC * sizeof(float));
    srand(42);
    for (size_t i = 0; i < maxA; i++) hA[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    for (size_t i = 0; i < maxB; i++) hB[i] = 2.0f * rand() / RAND_MAX - 1.0f;

    float *dA, *dB, *dC, *dRef;
    CK(cudaMalloc(&dA,   maxA * sizeof(float)));
    CK(cudaMalloc(&dB,   maxB * sizeof(float)));
    CK(cudaMalloc(&dC,   maxC * sizeof(float)));
    CK(cudaMalloc(&dRef, maxC * sizeof(float)));
    CK(cudaMemcpy(dA, hA, maxA * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, maxB * sizeof(float), cudaMemcpyHostToDevice));

    printf("%-7s %-8s %9s %9s %9s %8s\n",
           "shape", "kernel", "ms", "GFLOPS", "% cuBLAS", "error");
    for (int s = 0; s < NS; s++) {
        int N = shapes[s].N, K = shapes[s].K;
        size_t nC = (size_t)M * N;
        double flops = 2.0 * M * N * K;

        launch_cublas(dA, dB, nullptr, dRef, M, N, K);        // the right answer
        CK(cudaMemcpy(hRef, dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));

        float cublas_ms = 0.0f;
        for (int k = 0; k < NK; k++) {
            CK(cudaMemset(dC, 0, nC * sizeof(float)));
            float ms = time_ms(kernels[k].fn, dA, dB, dC, M, N, K);
            if (k == 0) cublas_ms = ms;
            CK(cudaMemcpy(hC, dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
            float err = rel_err(hRef, hC, nC);
            printf("%-7s %-8s %9.3f %9.0f %8.0f%% %8.1e%s\n",
                   shapes[s].name, kernels[k].name, ms, flops / ms / 1e6,
                   100.0f * cublas_ms / ms, err, err < 1e-4f ? "" : "  WRONG");
        }
        printf("\n");
    }

    cublasDestroy(g_handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}