# Oracle: llama.cpp

Сверка нашего эталона с llama.cpp на тех же весах.

## Сборка

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-cli llama-simple llama-tokenize
```

`logits.cpp` собирается отдельно и печатает top-10 логитов последней позиции
плюс пишет сырой массив в `/tmp/llama_logits.bin`:

```bash
g++ -O2 -std=c++17 logits.cpp \
  -I /tmp/llama.cpp/include -I /tmp/llama.cpp/ggml/include \
  -L /tmp/llama.cpp/build/bin \
  -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,/tmp/llama.cpp/build/bin -o /tmp/logits
```

## Сравнение

```bash
python3 compare_prompt.py "The capital of France is" models/LFM3-1.7B-Q4_K_M.gguf
```

Скрипт токенизирует промпт через `llama-tokenize`, считает логиты обеими
реализациями и печатает cosine, корреляцию и top-5.

Для численной проверки без эффекта квантования используйте F16-веса:
`models/LFM3-1.7B-F16.gguf`.
