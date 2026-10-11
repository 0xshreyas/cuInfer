#pragma once
// Register-tiled matmul kernels and a shape-based dispatcher.
// matmul_1d: 8 outputs per thread; matmul_2d: an 8x8 patch per thread; matmul_gpu picks per shape.

__global__ void matmul_kernel(const float *A, const float *B, const float *bias,
                              float *C, int M, int N, int K);   // naive, defined in model.cu

#define M1_BM 64
#define M1_BN 64
#define M1_BK 8
#define M1_TM 8

__global__ void matmul_1d(const float *A, const float *B, const float *bias,
                          float *C, int M, int N, int K) {
    __shared__ float As[M1_BM][M1_BK];
    __shared__ float Bs[M1_BK][M1_BN];
    int tid = threadIdx.x;
    int blockRow = blockIdx.y * M1_BM, blockCol = blockIdx.x * M1_BN;
    int threadCol = tid % M1_BN, threadRow = tid / M1_BN;
    int aRow = tid / M1_BK, aCol = tid % M1_BK;
    int bRow = tid / M1_BN, bCol = tid % M1_BN;
    float sum[M1_TM] = {0.0f};
    for (int t = 0; t < (K + M1_BK - 1) / M1_BK; t++) {
        int aR = blockRow + aRow, aC = t * M1_BK + aCol;
        As[aRow][aCol] = (aR < M && aC < K) ? A[aR * K + aC] : 0.0f;
        int bR = t * M1_BK + bRow, bC = blockCol + bCol;
        Bs[bRow][bCol] = (bR < K && bC < N) ? B[bR * N + bC] : 0.0f;
        __syncthreads();
        for (int k = 0; k < M1_BK; k++) {
            float b = Bs[k][threadCol];
            for (int i = 0; i < M1_TM; i++)
                sum[i] += As[threadRow * M1_TM + i][k] * b;
        }
        __syncthreads();
    }
    for (int i = 0; i < M1_TM; i++) {
        int row = blockRow + threadRow * M1_TM + i, col = blockCol + threadCol;
        if (row < M && col < N) {
            float v = sum[i];
            if (bias) v += bias[col];
            C[row * N + col] = v;
        }
    }
}

#define M2_BM 128
#define M2_BN 128
#define M2_BK 8
#define M2_TM 8
#define M2_TN 8
#define M2_NT ((M2_BM / M2_TM) * (M2_BN / M2_TN))

__global__ void matmul_2d(const float *A, const float *B, const float *bias,
                          float *C, int M, int N, int K) {
    __shared__ float As[M2_BM][M2_BK];
    __shared__ float Bs[M2_BK][M2_BN];
    int tid = threadIdx.x;
    int blockRow = blockIdx.y * M2_BM, blockCol = blockIdx.x * M2_BN;
    int threadRow = tid / (M2_BN / M2_TN), threadCol = tid % (M2_BN / M2_TN);
    float sum[M2_TM][M2_TN] = {{0.0f}};
    float aReg[M2_TM], bReg[M2_TN];
    for (int t = 0; t < (K + M2_BK - 1) / M2_BK; t++) {
        for (int idx = tid; idx < M2_BM * M2_BK; idx += M2_NT) {
            int r = idx / M2_BK, c = idx % M2_BK;
            int gr = blockRow + r, gc = t * M2_BK + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = tid; idx < M2_BK * M2_BN; idx += M2_NT) {
            int r = idx / M2_BN, c = idx % M2_BN;
            int gr = t * M2_BK + r, gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < M2_BK; k++) {
            for (int i = 0; i < M2_TM; i++) aReg[i] = As[threadRow * M2_TM + i][k];
            for (int j = 0; j < M2_TN; j++) bReg[j] = Bs[k][threadCol * M2_TN + j];
            for (int i = 0; i < M2_TM; i++)
                for (int j = 0; j < M2_TN; j++)
                    sum[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < M2_TM; i++) {
        int row = blockRow + threadRow * M2_TM + i;
        for (int j = 0; j < M2_TN; j++) {
            int col = blockCol + threadCol * M2_TN + j;
            if (row < M && col < N) {
                float v = sum[i][j];
                if (bias) v += bias[col];
                C[row * N + col] = v;
            }
        }
    }
}

#ifndef NAIVE_MAX_M
#define NAIVE_MAX_M 16   // up to this many tokens: naive
#endif
#define MIN_2D_BLOCKS 20   // 2d only if it makes at least this many blocks

void matmul_gpu(const float *A, const float *B, const float *bias,
                float *C, int M, int N, int K) {
    if (M <= NAIVE_MAX_M) {
        dim3 tpb(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_kernel<<<grid, tpb>>>(A, B, bias, C, M, N, K);
        return;
    }
    int blocks2d = ((N + M2_BN - 1) / M2_BN) * ((M + M2_BM - 1) / M2_BM);
    if (blocks2d >= MIN_2D_BLOCKS) {
        dim3 grid((N + M2_BN - 1) / M2_BN, (M + M2_BM - 1) / M2_BM);
        matmul_2d<<<grid, M2_NT>>>(A, B, bias, C, M, N, K);
    } else {
        dim3 grid((N + M1_BN - 1) / M1_BN, (M + M1_BM - 1) / M1_BM);
        matmul_1d<<<grid, M1_BM * M1_BN / M1_TM>>>(A, B, bias, C, M, N, K);
    }
}
