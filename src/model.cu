#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define MAGIC 0x63754C4D

__global__ void gelu_kernel(const float *in, float *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

__global__ void layerNorm_kernel(const float *in, float *out, const float *g, const float *b, int T, int d) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float *x = in + (size_t)row * d;
    float *y = out + (size_t)row * d;

    float s = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) s += x[i];
    sdata[tid] = s;
    __syncthreads();

    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    float mean = sdata[0] / d;
    __syncthreads();
    
    float v = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = x[i] - mean;
        v += diff * diff;
    }
    sdata[tid] = v;
    __syncthreads();
    
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    float var = sdata[0] / d;
    __syncthreads();

    float inv = rsqrtf(var + 1e-5f);
    for (int i = tid; i < d; i += blockDim.x) {
        y[i] = g[i] * (x[i] - mean) * inv + b[i];
    }
}

__global__ void sum_kernel(const float *in, float *block_sums, size_t n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    //Grid-Stride Loop
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for(; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

__global__ void softmax_kernel(float *x, int n_rows, int row_len) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float *row = x + (size_t)blockIdx.x * row_len;

    // Row max
    float mx = -INFINITY;
    for (int i = tid; i < row_len; i += blockDim.x) mx = fmaxf(mx, row[i]);
    sdata[tid] = mx;
    __syncthreads();

    for (int k = blockDim.x/2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] = fmaxf(sdata[tid], sdata[tid + k]);
        __syncthreads();
    }

    float rowmax = sdata[0];
    __syncthreads();

    // Exp and Sum
    float s = 0.0f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float e = expf(row[i] - rowmax);
        row[i] = e;
        s += e;
    }

    sdata[tid] = s;
    __syncthreads();

    for (int k = blockDim.x/2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    float total = sdata[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < row_len; i += blockDim.x) row[i] /= total;
}

__global__ void matmul_kernel(const float *A, const float *B, const float *bias, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + (bias ? bias[col] : 0.0f);
    }
}

__global__ void add_kernel(float *x, const float *y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

__global__ void embed_kernel(const int *tokens, const float *wte, const float *wpe, float *out, int T, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * d) {
        int t = i / d;
        int col = i % d;
        out[i] = wte[(size_t)tokens[t] * d + col] + wpe[(size_t)t * d + col];
    }
}

__global__ void scores_kernel(const float *qkv, float *scores, int T, int n_head, int head_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_head * T * T) return;
    int h = idx / (T * T);
    int i = (idx / T) % T;
    int j = (idx % T);

    int d3 = 3 * n_head * head_dim;
    const float *q = qkv + (size_t)i * d3 + 0 + h * head_dim;
    const float *k = qkv + (size_t)j * d3 + n_head * head_dim + h * head_dim;

    float sum = 0.0f;
    for (int e = 0; e < head_dim; e++) sum+= q[e] * k[e];
    scores[(size_t)h * T * T + i * T + j] = sum * rsqrtf((float)head_dim);
}

__global__ void mask_kernel(float *scores, int T, int n_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_head * T * T) return;

    int h = idx / (T * T);
    int i = (idx / T) % T;
    int j = (idx % T);

    if (j > i) scores[(size_t)h * T * T + i * T + j] = -INFINITY;
}

__global__ void weighted_v_kernel(const float *weights, const float *qkv, float *out, int T, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * n_head * head_dim) return;
    int t = idx / (n_head * head_dim);
    int h = (idx / head_dim) % n_head;
    int e = (idx % head_dim);

    float sum = 0.0f;
    int d3 = 3 * n_head * head_dim;
    for (int j = 0; j < T; j++) {
        sum += weights[h * T * T + t * T + j] * qkv[j * d3 + 2 * n_head * head_dim + h * head_dim + e];
    }
    out[t * n_head * head_dim + h * head_dim + e] = sum;
}
// Structure for the loader includes the 12 tensors in a layer and parameters of a model
typedef struct {
    float *ln_1_w, *ln_1_b;
    float *attn_w, *attn_b;
    float *proj_w, *proj_b;
    float *ln_2_w, *ln_2_b;
    float *fc_w, *fc_b;
    float *fcproj_w, *fcproj_b;
} Layer;

typedef struct {
    int n_layer, n_head, d_model, vocab_size, max_pos;
    float *all;
    float *wte, *wpe;
    Layer layers[12];
    float *ln_f_w, *ln_f_b;
} Model;

static float *TAKE(Model *m, size_t *off, size_t n) {
        float *p = m->all + *off;
        *off += n;
        return p;
    }

int model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        printf("Could not open the file\n");
        return 1;
    }

    int header[7];
    if (fread(header, sizeof(int), 7, f) != 7) {
        printf("Failed to read the header\n");
        fclose(f);
        return 1;
    }
    if (header[0] != MAGIC) {
        printf("Bad Magic: Got 0x%X, expected 0x%X\n", header[0], MAGIC);
        fclose(f);
        return 1;
    }
    m->n_layer = header[2];
    m->n_head = header[3];
    m->d_model = header[4];
    m->vocab_size = header[5];
    m->max_pos = header[6];

    printf("n_layer = %d, heads = %d, d_model = %d, vocab_size = %d, max_pos = %d\n", m->n_layer, m->n_head, m->d_model, m->vocab_size, m->max_pos);
    size_t n_floats = 124439808;
    float *host_buffer = (float*)malloc(n_floats * sizeof(float));
    if (host_buffer == NULL) {
        printf("Out of memory\n");
        fclose(f);
        return 1;
    }
    size_t data_received = fread(host_buffer, sizeof(float), n_floats, f);
    if (data_received != n_floats) {
        printf("Short Read: Got %zu out of %zu floats\n", data_received, n_floats);
        fclose(f);
        return 1;
    }
    fclose(f);
    cudaError_t e = cudaMalloc(&m->all, n_floats * sizeof(float));
    if (e != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(e));
        free(host_buffer);
        return 1;
    }
    e = cudaMemcpy(m->all, host_buffer, n_floats * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(e));
        free(host_buffer);
        return 1;
    }
    free(host_buffer);
    size_t d = m->d_model;
    size_t off = 0;
    m->wte = TAKE(m, &off, (size_t)m->vocab_size * m->d_model);
    m->wpe = TAKE(m, &off, (size_t)m->max_pos * m->d_model);

    for (int i = 0; i < m->n_layer; i++) {
        Layer *L = &m->layers[i];
        L->ln_1_w = TAKE(m, &off, d);
        L->ln_1_b = TAKE(m, &off, d);
        L->attn_w = TAKE(m, &off, d * 3 * d);
        L->attn_b = TAKE(m, &off, 3 * d);
        L->proj_w   = TAKE(m, &off, d * d);
        L->proj_b   = TAKE(m, &off, d);
        L->ln_2_w   = TAKE(m, &off, d);
        L->ln_2_b   = TAKE(m, &off, d);
        L->fc_w     = TAKE(m, &off, d * 4 * d);
        L->fc_b     = TAKE(m, &off, 4 * d);
        L->fcproj_w = TAKE(m, &off, 4 * d * d);
        L->fcproj_b = TAKE(m, &off, d);
    }
    m->ln_f_w = TAKE(m, &off, d);
    m->ln_f_b = TAKE(m, &off, d);
    if (off != n_floats) {
        printf("layout error: used %zu of %zu floats\n", off, n_floats);
        return 1;
    }
    return 0;
}

void model_free(Model *m) {
    cudaFree(m->all);
}

// To Verify the data from the GPU side, first method which is Copy back & check
void peek(const char *label, float *ptr, int n) {
    float host[8];
    cudaMemcpy(host, ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%s: ", label);
    for (int i = 0; i < n; i++) printf("%.8f ", host[i]);
    printf("\n");
}

double sum_on_gpu(const float *ptr, size_t n) {
    int blocks = 256;
    int threads = 256;
    float *d_block_sums;
    cudaError_t e = cudaMalloc(&d_block_sums, blocks * sizeof(float));
    if (e != cudaSuccess) {
        printf("Error from CUDA: %s\n", cudaGetErrorString(e));
        return -1.0;
    }
    sum_kernel<<<blocks, threads>>>(ptr, d_block_sums, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error from CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float *host_array = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(host_array, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += host_array[i];
    free(host_array);
    cudaFree(d_block_sums);
    return sum; 
    
}

void test_gelu() {
    size_t n = 10000;
    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    float *h_ref = (float*)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        h_in[i] = (rand() % 2000 - 1000) / 100.0f;
    }

    // CPU reference
    for(size_t i = 0; i < n; i++) {
        float x = h_in[i];
        h_ref[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    gelu_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(h_out[i] - h_ref[i]);
        if (d > maxDiff) maxDiff = d;
    }

    printf("GELU Max Diff: %.3e %s\n", maxDiff, maxDiff < 1e-5 ? "PASS" : "FAIL");
    free(h_in);
    free(h_out);
    free(h_ref);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_layernorm() {
    int T = 5, d = 768;
    size_t n = (size_t)T * d;

    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    float *h_ref = (float*)malloc(n * sizeof(float));
    float *h_g = (float*)malloc(d * sizeof(float));
    float *h_b = (float*)malloc(d * sizeof(float));

    for (size_t i = 0; i < n; i++) h_in[i] = (rand() % 2000 - 1000) / 100.0f;
    for (int i = 0; i < d; i++) {
        h_g[i] = 1.0f + (rand() % 100) / 1000.0f;
        h_b[i] = (rand() % 100 - 50) / 1000.0f;
    }

    // CPU reference
    for (int r = 0; r < T; r++) {
        const float *x = h_in + (size_t)r * d;
        float *y = h_ref + (size_t)r * d;

        double sum = 0.0;
        for (int i = 0; i < d; i++) sum += x[i];
        double mean = sum / d;

        double vs = 0.0;
        for (int i = 0; i < d; i++) { double diff = x[i] - mean; vs += diff * diff; }
        double var = vs / d;

        double inv = 1.0 / sqrt(var + 1e-5);
        for (int i = 0; i < d; i++) y[i] = h_g[i] * (x[i] - mean) * inv + h_b[i];
    }

    float *d_in, *d_out, *d_g, *d_b;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_g, d * sizeof(float));
    cudaMalloc(&d_b, d * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g,  d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b,  d * sizeof(float), cudaMemcpyHostToDevice);

    layerNorm_kernel<<<T, 256>>>(d_in, d_out, d_g, d_b, T, d);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { 
        printf("kernel error: %s\n", cudaGetErrorString(e)); 
        return; 
    }

    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxdiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float df = fabsf(h_out[i] - h_ref[i]);
        if (df > maxdiff) maxdiff = df;
    }
    printf("layerNorm Max Diff: %.3e  %s\n", maxdiff, maxdiff < 1e-4 ? "PASS" : "FAIL");

    free(h_in); 
    free(h_out); 
    free(h_ref); 
    free(h_g); 
    free(h_b);
    cudaFree(d_in); 
    cudaFree(d_out); 
    cudaFree(d_g); 
    cudaFree(d_b);
}

void softmax_test() {
    int n_rows = 4, row_len = 100;
    size_t n = (size_t)n_rows * row_len;

    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    float *h_ref = (float*)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        h_in[i] = (rand() % 2000 - 1000) / 100.0f;
    }

    // CPU reference
    for (int r = 0; r < n_rows; r++) {
        const float *x = h_in + (size_t)r * row_len;
        float *y = h_ref + (size_t)r * row_len;

        float mx = -INFINITY;
        for (int i = 0; i < row_len; i++) if (x[i] > mx) mx = x[i];

        double sum = 0.0;
        for (int i = 0; i < row_len; i++) {
            y[i] = expf(x[i] - mx);
            sum += y[i];
        }
        for (int i = 0; i < row_len; i++) y[i] /= sum;
    }

    float *d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<n_rows, 256>>>(d_x, n_rows, row_len);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Kernel Error: %s\n", cudaGetErrorString(e)); 
        return;
    }
    cudaMemcpy(h_out, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float df = fabsf(h_out[i] - h_ref[i]);
        if (df > maxDiff) maxDiff = df;
    }

    float worst_rowsum = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        double s = 0.0;
        for (int i = 0; i < row_len; i++) {
            s += h_out[(size_t)r * row_len + i];
        }
        float df = fabsf((float)s - 1.0f);
        if (df > worst_rowsum) worst_rowsum = df;
    }
    printf("SoftMax Max Diff: %.3e RowSum Err: %.3e %s\n", maxDiff, worst_rowsum, (maxDiff < 1e-5 && worst_rowsum < 1e-5) ? "PASS" : "FAIL");
    free(h_in);
    free(h_out);
    free(h_ref);
    cudaFree(d_x);
}

void mlp_forward(Layer *L, float *x, float *hidden, float *out, int T, int d) {
    dim3 tpb(16, 16);
    
    dim3 g1((4*d + 15)/16, (T + 15)/16);
    matmul_kernel<<<g1, tpb>>>(x, L->fc_w, L->fc_b, hidden, T, 4*d, d);

    size_t hn = (size_t)T * 4 * d;
    gelu_kernel<<<(hn + 255)/256, 256>>>(hidden, hidden, hn);

    dim3 g2((d + 15)/16, (T + 15)/16);
    matmul_kernel<<<g2, tpb>>>(hidden, L->fcproj_w, L->fcproj_b, out, T, d, 4*d);
}

void test_embed(Model *m) {
    int T = 5, d = 768;
    size_t n = (size_t)T * d;
    int h_tokens[5];
    FILE *f = fopen("../export/oracle_tokens.bin", "rb");
    if (!f) {
        printf("Could not open oracle_tokens.bin\n");
        return;
    }
    if(fread(h_tokens, sizeof(int), T, f) != (size_t)T) {
        printf("Short read on oracle_tokens.bin\n");
        fclose(f);
        return;
    }
    fclose(f);
    float *h_ref = (float*)malloc(n * sizeof(float));
    f = fopen("../export/oracle_embed.bin", "rb");
    if (!f) {
        printf("could not open oracle_embed.bin\n");
        return;
    }
    if(fread(h_ref, sizeof(float), n, f) != n) {
        printf("Short read on oracle_embed.bin\n");
        fclose(f);
        free(h_ref);
        return;
    }
    fclose(f);

    int *d_tokens;
    float *d_out;
    cudaMalloc(&d_tokens, T * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_tokens, h_tokens, T * sizeof(int), cudaMemcpyHostToDevice);
    embed_kernel<<<(n + 255)/256, 256>>>(d_tokens, m->wte, m->wpe, d_out, T, d);
    cudaDeviceSynchronize();

    float *h_out = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float df = fabsf(h_out[i] - h_ref[i]);
        if (df > maxDiff) maxDiff = df;
    }

    printf("Embed Difference: %.3e %s\n", maxDiff, maxDiff < 1e-4 ? "PASS" : "FAIL");
    free(h_ref);
    free(h_out);
    cudaFree(d_tokens);
    cudaFree(d_out);    
}

void test_scores() {
    size_t n = 5 * 2304;
    float *h_qkv = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) h_qkv[i] = (rand() % 2000 - 1000) / 100.0f;
    float *d_qkv;
    cudaMalloc(&d_qkv, n * sizeof(float));
    cudaMemcpy(d_qkv, h_qkv, n * sizeof(float), cudaMemcpyHostToDevice);

    size_t out_n = (size_t)12 * 5 * 5;
    float *d_scores;
    cudaMalloc(&d_scores, out_n * sizeof(float));

    scores_kernel<<<(out_n + 255)/256, 256>>>(d_qkv, d_scores, 5, 12, 64);
    cudaDeviceSynchronize();

    float *h_out = (float*)malloc(out_n * sizeof(float));
    cudaMemcpy(h_out, d_scores, out_n * sizeof(float), cudaMemcpyDeviceToHost);

    float *h_ref = (float*)malloc(out_n * sizeof(float));
    int T = 5, n_head = 12, head_dim = 64, d3 = 3 * n_head * head_dim;
    for (int h = 0; h < n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                const float *q = h_qkv + (size_t)i * d3 + h * head_dim;
                const float *k = h_qkv + (size_t)j * d3 + n_head * head_dim + h * head_dim;

                float sum = 0.0f;
                for (int e = 0; e < head_dim; e++) sum+= q[e] * k[e];
                h_ref[h * T * T + i * T + j] = sum / sqrtf((float)head_dim);
            }
        }
    }

    float maxDiff = 0.0f;
    for (size_t i = 0; i < out_n; i++) {
        float df = fabsf(h_out[i] - h_ref[i]);
        if (df > maxDiff) maxDiff = df;
    }

    printf("Scores Difference: %.3e %s\n", maxDiff, maxDiff < 1e-4 ? "PASS" : "FAIL");
    cudaFree(d_qkv);
    cudaFree(d_scores);
    free(h_qkv);
    free(h_ref);
    free(h_out);
}

void attention_forward(Layer *L, float *x, float *qkv, float *scores, float *attn_out, float *out, int T, int d, int n_head) {
    int head_dim = d / n_head;
    dim3 tpb(16, 16);

    // QKV Projection: (T,d) @ (d,3d) -> (T,3d)
    dim3 g1((3 * d + 15)/16, (T + 15)/16);
    matmul_kernel<<<g1, tpb>>>(x, L->attn_w, L->attn_b, qkv, T, 3 * d, d);

    // Scores
    int n_scores = n_head * T * T;
    scores_kernel<<<(n_scores + 255)/256, 256>>>(qkv, scores, T, n_head, head_dim);

    // Casual Mask
    mask_kernel<<<(n_scores + 255)/256, 256>>>(scores, T, n_head);

    // Softmax over each row of length T
    softmax_kernel<<<n_head * T, 256>>>(scores, n_head * T, T);

    // Weighted V
    int n_out = T * d;
    weighted_v_kernel<<<(n_out + 255)/256, 256>>>(scores, qkv, attn_out, T, n_head, head_dim);

    // Output Projection (T,d) @ (d,d) -> (T,d)
    dim3 g2((d + 15)/16, (T + 15)/16);
    matmul_kernel<<<g2, tpb>>>(attn_out, L->proj_w, L->proj_b, out, T, d, d);
}

void block_forward(Layer *L, float *x, float *ln_out, float *qkv, float *scores, float *attn_out, float *hidden, float *tmp, int T, int d, int n_head) {
    size_t n = (size_t)T * d;
    int blocks = (n + 255) / 256;
    layerNorm_kernel<<<T, 256>>>(x, ln_out, L->ln_1_w, L->ln_1_b, T, d);
    attention_forward(L, ln_out, qkv, scores, attn_out, tmp, T, d, n_head);
    add_kernel<<<blocks, 256>>>(x, tmp, n);
    layerNorm_kernel<<<T, 256>>>(x, ln_out, L->ln_2_w, L->ln_2_b, T, d);
    mlp_forward(L, ln_out, hidden, tmp, T, d);
    add_kernel<<<blocks, 256>>>(x, tmp, n);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <weights.bin>\n", argv[0]);
        return 1;
    }

    Model m;
    if (model_load(&m, argv[1]) != 0) {
        printf("Load Failed\n");
        return 1;
    }

    printf("Load Success\n");
    printf("WTE at offset 0, ln_f_b at %td floats \n", m.ln_f_b - m.all);
    // Copy back & check method on wte and ln_f_b
    peek("wte", m.wte, 3);
    peek("ln_f_b", m.ln_f_b, 3);
    size_t wte_n = (size_t)m.vocab_size * m.d_model;
    double s = sum_on_gpu(m.wte, wte_n);
    printf("WTE Sum: %.6f (comparing against python's checksum which is =14659.962891=)\n", s);
    test_gelu();
    test_layernorm();
    softmax_test();
    test_embed(&m);
    test_scores();
    model_free(&m);
    return 0;
}