import subprocess, sys, os
import numpy as np

LLAMA = os.environ.get('LLAMA_CPP', '/tmp/llama.cpp')
LOGITS_BIN = '/tmp/logits'
TOKENIZE = os.path.join(LLAMA, 'build/bin/llama-tokenize')
OURS = './target/release/lfm-bench'
LLAMA_LOGITS = '/tmp/llama_logits.bin'
OUR_LOGITS = '/tmp/our_logits.bin'


def tokenize(model, prompt):
    out = subprocess.run([TOKENIZE, '-m', model, '-p', prompt],
                         capture_output=True, text=True)
    ids = []
    for line in out.stdout.splitlines():
        if ' -> ' in line:
            ids.append(line.split(' -> ')[0].strip())
    return ','.join(ids)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'
    model = sys.argv[2] if len(sys.argv) > 2 else 'models/LFM3-1.7B-Q4_K_M.gguf'
    ids = tokenize(model, prompt)
    if not ids:
        raise SystemExit('tokenize failed')
    subprocess.run([LOGITS_BIN, model, prompt], capture_output=True, check=True)
    subprocess.run([OURS, 'ref-debug', '--model', model, '--ids', ids, '--out', OUR_LOGITS],
                   capture_output=True, check=True)
    a = np.fromfile(LLAMA_LOGITS, dtype=np.float32)
    b = np.fromfile(OUR_LOGITS, dtype=np.float32)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f'model  {model}')
    print(f'prompt {prompt!r} ids {ids}')
    print(f'cosine {cos:+.6f}  corr {corr:+.4f}')
    print(f'llama  top5 {np.argsort(-a)[:5].tolist()}')
    print(f'ours   top5 {np.argsort(-b)[:5].tolist()}')
    print(f'max|dlogit| {float(np.max(np.abs(a - b))):.4f}')


if __name__ == '__main__':
    main()
