#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf \"prompt\" [n]\n", argv[0]); return 1; }
    const char * path = argv[1];
    std::string prompt = argv[2];
    int n_max = argc > 3 ? atoi(argv[3]) : 16;

    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, true, true);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    int nv = llama_vocab_n_tokens(vocab);
    int pos = n;
    std::vector<llama_token> out;
    for (int i = 0; i < n_max; ++i) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { fprintf(stderr, "no logits\n"); return 1; }
        int best = 0;
        for (int j = 1; j < nv; ++j) {
            if (lg[j] > lg[best]) best = j;
        }
        out.push_back(best);
        if (best == llama_vocab_eos(vocab)) break;
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = best;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        llama_batch_free(batch);
    }
    printf("ids: [");
    for (size_t i = 0; i < out.size(); ++i) {
        printf("%s%d", i ? ", " : "", out[i]);
    }
    printf("]\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
