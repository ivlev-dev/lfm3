#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [prompt] [n]\n", argv[0]); return 1; }
    const char * path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    int n_max = argc > 3 ? atoi(argv[3]) : 32;

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512, cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, true, true);
    printf("tokens:");
    for (auto t : toks) printf(" %d", t);
    printf("\n");
    if (n > n_max) n = n_max;
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    int nv = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, n - 1);
    if (!lg) { fprintf(stderr, "no logits\n"); return 1; }
    std::vector<int> idx(nv);
    for (int j = 0; j < nv; ++j) idx[j] = j;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    printf("top10:");
    for (int k = 0; k < 10; ++k) printf(" %d=%.4f", idx[k], lg[idx[k]]);
    printf("\n");
    FILE * f = fopen("/tmp/llama_logits.bin", "wb");
    fwrite(lg, sizeof(float), nv, f);
    fclose(f);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
