#include<stdio.h>
#include<stdlib.h>

#define MAGIC 0x63754C4D

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <weights.bin>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("Could not open %s\n", argv[1]);
        return 1;
    }

    int header[7];
    if (fread(header, sizeof(int), 7, f) != 7) {
        printf("Failed to read header\n");
        fclose(f);
        return 1;
    }

    if (header[0] != MAGIC) {
        printf("Bad Magic: Got 0x%X, expected 0x%X\n", header[0], MAGIC);
        fclose(f);
        return 1;
    }

    int version    = header[1];
    int n_layer    = header[2];
    int n_head     = header[3];
    int d_model    = header[4];
    int vocab_size = header[5];
    int max_pos    = header[6];

    printf("version=%d, layers=%d, heads=%d, d_model=%d, vocab_size=%d, max_pos=%d", version, n_layer, n_head, d_model, vocab_size, max_pos);

    size_t n_floats = 124439808;
    float *weights = (float*)malloc(n_floats * sizeof(float));
    if (weights == NULL) {
        printf("Out of memory\n");
        fclose(f);
        return 1;
    }

    size_t got = fread(weights, sizeof(float), n_floats, f);
    if (got != n_floats) {
        printf("Short Read: Got %zu of %zu floats\n", got, n_floats);
        free(weights);
        fclose(f);
        return 1;
    }
    fclose(f);

    // Checksum wte the first tensor

    size_t wte_size = (size_t)vocab_size * d_model;
    float *wte = weights;
    printf("\nwte first 3: %.8f %.8f %.8f\n", wte[0], wte[1], wte[2]);

    double sum = 0.0;
    for(size_t i = 0; i < wte_size; i++) sum += wte[i];
    printf("\nWTE Sum: %.6f\n", sum);

    // Checksum ln_f.bias, the last 768 floats of the file
    float *ln_f_bias = weights + n_floats - d_model;
    printf("\nln_f.bias first 3: %.8f %.8f %.8f\n", ln_f_bias[0], ln_f_bias[1], ln_f_bias[2]);

    sum = 0.0;
    for (int i = 0; i < d_model; i++) sum += ln_f_bias[i];
    printf("\nln_f.bias sum %.6f\n", sum);
    
    free(weights);
    return 0;
}