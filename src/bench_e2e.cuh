// End-to-end timing of the whole model: CUDA events, warm-up, averaged runs.
// prefill = one forward pass over a T-token prompt (time to first token)
// decode  = tokens per second when generating one token at a time with the KV cache
void bench_e2e(Model *m) {
    const int d = m->d_model, V = m->vocab_size;
    const int MAXT = 1024, TMAX = 512;
    int *d_tok;
    float *logits, *x, *ln_out, *qkv, *scores, *attn_out, *hidden, *tmp, *kc, *vc;
    cudaMalloc(&d_tok,    TMAX * sizeof(int));
    cudaMalloc(&logits,   (size_t)TMAX * V * sizeof(float));
    cudaMalloc(&x,        (size_t)TMAX * d * sizeof(float));
    cudaMalloc(&ln_out,   (size_t)TMAX * d * sizeof(float));
    cudaMalloc(&qkv,      (size_t)TMAX * 3 * d * sizeof(float));
    cudaMalloc(&scores,   (size_t)12 * MAXT * MAXT * sizeof(float));
    cudaMalloc(&attn_out, (size_t)TMAX * d * sizeof(float));
    cudaMalloc(&hidden,   (size_t)TMAX * 4 * d * sizeof(float));
    cudaMalloc(&tmp,      (size_t)TMAX * d * sizeof(float));
    cudaMalloc(&kc,       (size_t)12 * MAXT * d * sizeof(float));
    cudaMalloc(&vc,       (size_t)12 * MAXT * d * sizeof(float));

    int h_tok[TMAX];                                  // any valid token ids work for timing
    for (int i = 0; i < TMAX; i++) h_tok[i] = (i * 7919) % V;
    cudaMemcpy(d_tok, h_tok, TMAX * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;
    printf("\n=== End-to-end (clean timing) ===\n");

    // prefill: 3 warm-ups, then the average of 10 runs
    int Ts[3] = {5, 128, 512};
    for (int i = 0; i < 3; i++) {
        int T = Ts[i];
        for (int w = 0; w < 3; w++)
            forward_pass(m, d_tok, logits, x, ln_out, qkv, scores, attn_out,
                         hidden, tmp, kc, vc, 0, T, MAXT);
        cudaEventRecord(s);
        for (int r = 0; r < 10; r++)
            forward_pass(m, d_tok, logits, x, ln_out, qkv, scores, attn_out,
                         hidden, tmp, kc, vc, 0, T, MAXT);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("prefill T=%3d : %8.3f ms\n", T, ms / 10);
    }

    // decode: a 128-token prompt fills the cache, then 128 new tokens one at a time
    const int P = 128, NDEC = 128;
    forward_pass(m, d_tok, logits, x, ln_out, qkv, scores, attn_out,
                 hidden, tmp, kc, vc, 0, P, MAXT);
    for (int w = 0; w < 8; w++)                       // warm-up steps
        forward_pass(m, d_tok, logits, x, ln_out, qkv, scores, attn_out,
                     hidden, tmp, kc, vc, P + w, 1, MAXT);
    cudaEventRecord(s);
    for (int i = 0; i < NDEC; i++)
        forward_pass(m, d_tok, logits, x, ln_out, qkv, scores, attn_out,
                     hidden, tmp, kc, vc, P + i, 1, MAXT);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    printf("decode        : %8.3f ms/token = %6.1f tokens/s\n",
           ms / NDEC, 1000.0f * NDEC / ms);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error in bench_e2e: %s\n", cudaGetErrorString(err));
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_tok); cudaFree(logits); cudaFree(x); cudaFree(ln_out); cudaFree(qkv);
    cudaFree(scores); cudaFree(attn_out); cudaFree(hidden); cudaFree(tmp);
    cudaFree(kc); cudaFree(vc);
}
