#include<stdio.h>
#include<stdlib.h>
#define MAGIC 0x63754C4D

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
    model_free(&m);
    return 0;
}