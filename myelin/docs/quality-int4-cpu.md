# Качество int4 на CPU-эталоне: что именно ломает текст

Документ отвечает на вопрос «какое минимальное изменение формата int4-весов
перестаёт вырождать текст». Все замеры сделаны на **CPU-пути** (`CpuBackend`,
точная f32-арифметика) без телефона, на чекпойнте
`models/LFM3-1.7B-Q4_K_M.gguf`. Код DSP, `myelin-npu` и `backend-opencl` не
менялся.

Экспериментальные ключи (`MYELIN_CPU_*`) — временные, читаются один раз при
загрузке чекпойнта и не являются прод-форматом.

## 1. Что какой флаг квантует (по коду)

Важно для однозначности: в CPU-пути **KV-кэш хранится в f32** (`kv.rs`,
`PagedKvCache`), активации — f32. Поэтому старый `MYELIN_CPU_INT4=1` трогает
**только матрицы весов** и ничего больше.

| Флаг | Что делает | Где в коде |
|---|---|---|
| `MYELIN_CPU_INT4=1` | симметричный int4, `absmax/7`, **один масштаб на строку** | `prepare_weights` → `quantize_row_blocks` |
| `MYELIN_CPU_WBLOCK=32\|64\|128\|256\|row` | тот же int4, масштаб на блок; `row` = поведение `INT4` | там же |
| `MYELIN_CPU_WASYM=1` | асимметричный int4 с min-термом: `w = step·q + min`, `q∈0..15`, масштаб и min на блок (`WBLOCK`) | `quantize_row_blocks` |
| `MYELIN_CPU_WHAD=32\|128` | нормализованное блочное преобразование Адамара строк весов и входа (`W·H`, `H·x`) до квантования | `hadamard_blocks` + `Matrix::multiply` |
| `MYELIN_CPU_Q4K=1` | не переквантовывать: значения как в GGUF (в этом чекпойнте — нативный Q4_K) | `WeightOpts::from_env` |
| `MYELIN_CPU_KVK/KVV=f32\|int8\|int4` | квантование K/V при записи в кэш; K — масштаб на `head_dim` (как DSP), V — на вектор токена (упрощение) | `attention_layer` → `quantize_symmetric` |
| `MYELIN_CPU_INT8_ACT=1` | int8-активации, `absmax/110`, как в DSP (существующий флаг) | `maybe_quantize_i8` |
| `MYELIN_CPU_CONV16=1` | матрицы `shortconv.*` — int16 per-row вместо int4 (CPU-аналог Y16, не бит-в-бит) | `prepare_weights` |
| `MYELIN_CPU_WHF32=1` | не квантовать `token_embd`/`output` (диагностика головы) | `prepare_weights` |

Список матриц под int4: `token_embd.weight`, `output.weight` (если не tied),
`blk.N.ffn_{up,gate,down}`, `blk.N.attn_{q,k,v,gate,output}`,
`blk.N.shortconv.{in_proj,conv,out_proj}`. Нормы (`*_norm.weight`) не
квантуются.

Метрики в выводе `wstats`:

* `cos` — косинус между исходными и восстановленными весами, по всем
  элементам (доминируется крупными весами);
* `relrmse = ||w − ŵ|| / ||w||` — не маскируется выбросами, именно он
  показывает слишком грубый per-row масштаб;
* `bytes/weight` — полезная нагрузка: int4 0.5 Б + масштабы (f32) на группу.

Промпты: P1 `The capital of France is`, P2 `The largest planet in the Solar
System is`; greedy, 24/40 токенов (P2 — 30). Оптимизация release.

## 2. Основная матрица (P1, 24 токена, если не указано иное)

| # | Конфигурация | cos | relRMSE | Б/вес | Текст (начало) |
|---|---|---|---|---|---|
| 0 | Q4_K нативный (`Q4K=1`, = f32-эталон) | 1.00000 | 0.0000 | 0.5625 | ` Paris, and the capital of Japan is Tokyo. The capital of Brazil is Brasília,` |
| a | int4 per-row (`INT4=1`) | 0.98340 | 0.1843 | 0.5020 | ` the country, and the country is the **United States** of **the** **the** **the**` |
| b | int4 блок 256 | 0.99048 | 0.1390 | 0.5156 | ` Paris, and the country has 109,000 inhabitants. The population of France is about 68,000,000` |
| c | int4 блок 128 | 0.99212 | 0.1263 | 0.5312 | ` Paris, and it is located in the north of the country. Paris is known for its culture, fashion, and history` |
| d | int4 блок 32 | 0.99492 | 0.1010 | 0.6250 | ` Paris, and it is known for its culture, history, and landmarks. Paris is also a major tourist destination` |
| e | блок 256 + H32 | 0.99209 | 0.1266 | 0.5156 | ` Paris, and the capital of Japan is Tokyo. The capital of Brazil is Brasília. The capital of the United States` |
| f | блок 128 + H32 | 0.99320 | 0.1173 | 0.5312 | ` Paris, the capital of France is Paris, Paris is the capital of France,` — цикл |
| g | блок 32 + H32 | 0.99539 | 0.0966 | 0.6250 | ` Paris, the capital of France is Paris, Paris is the capital of France,` — цикл |
| — | **per-row + H32** | 0.98824 | 0.1547 | **0.5020** | ` Paris. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of the United States is Washington, D.C.` (40 ток.) |
| — | per-row + H128 | 0.98839 | 0.1537 | 0.5020 | ` the city of the capital of the capital of the capital of …` — жёсткий цикл |
| — | блок 1024 | 0.98507 | 0.1746 | 0.5039 | ` on the Bay of the…………………………………………` — коллапс |
| — | блок 512 | 0.98864 | 0.1520 | 0.5078 | ` Paris, and the capital of the United States is Washington, D.C., … the District of Columbia` — уходит в цикл к ~20 токену |
| h | asym per-row (`WASYM=1`) | 0.98726 | 0.1612 | 0.5039 | ` the City of Paris and it is Paris. The capital of France is Paris and I am Paris` |
| — | asym блок 256 | 0.99332 | 0.1162 | 0.5312 | ` Paris, the capital of the country located in Western Europe, and the most populous city in France` |
| — | asym блок 128 | 0.99473 | 0.1030 | 0.5625 | ` Paris, and the capital of the United States is Washington, D.C., a district consisting of …` |
| i1 | блок 256 + H32 + CONV16 | 0.99314 | 0.1178 | 0.5156 | ` Paris. The capital of Japan is Tokyo. The capital of Germany is Berlin. The capital of the United States is Washington` |
| i2 | per-row + H32 + CONV16 | 0.98973 | 0.1445 | 0.5020 | ` Paris. The capital of the capital of France is the capital of France …` — цикл (см. §5) |
| — | блок 64 | 0.99359 | 0.1137 | 0.5625 | ` the city of Paris. The capital of the United States is the city of Washington, D.C.` — первый токен неверный |
| — | блок 256 + голова f32 | 0.99056 | — | 0.5156 | ` Paris, and the country has a population of about 68 million people. …` |
| — | per-row + голова f32 | 0.98333 | — | 0.5020 | ` the "Capital of France" or "France" depending on the level of the article we are on` — всё ещё плохо |

Ключевые наблюдения:

1. **Виноват per-row масштаб (2048 значений на один absmax), а не int4 как
   таковой.** Блок 256/128/32 даёт связный текст; 512 — на границе, 1024 и
   per-row — коллапс. Переход резкий и воспроизводимый.
2. **Связность текста не монотонна по cos:** per-row+H32 (`cos 0.98824`)
   даёт связный текст, а блок 128+H32 (`cos 0.99320`) — цикл; `CONV16`
   поднимает cos (0.98824 → 0.98973) и ломает текст. **cos(W) непригоден как
   критерий приёмки**; relRMSE лучше (коллапс при ≳0.16, связный текст при
   ≲0.14), но тоже не достаточен.
3. **per-row + H32 — единственный вариант с нулевым оверхедом по байтам.**
   Ротация «размазывает» выбросы внутри строки, и один absmax на строку
   становится эквивалентен блочному. H128 при per-row, наоборот, даёт цикл.
4. **Min-терм не обязателен.** asym per-row всё равно ломает первый токен
   (` the City…`), asym блок 256 даёт лучший relRMSE (0.1162) и отличный текст,
   но симметричный блок 256 уже достаточен. Min-терм стоит брать только если
   ядро кладёт bias бесплатно (HMX `input_bias`).
5. **Голова/эмбеддинги — не главная причина.** per-row по телу при f32-голове
   всё равно вырождается, а блок 256 по телу при f32-голове — связный.
   Держать голову в f32 не нужно.

## 3. Изоляция KV (приоритет заказчика)

Веса — нативные (Q4_K), меняется только кэш (P1, 24 токена):

| K / V | Текст | Вывод |
|---|---|---|
| f32 / f32 | ` Paris, and the capital of Japan is Tokyo. The capital of Brazil is Brasília, …` | эталон |
| f32 / **int4** | ` Paris, and the capital of the United States is Washington, D.C. The capital of Canada is Ottawa. …` | связно, уезжает тема |
| **int4** / f32 | ` Paris, and the capital of Japan is Tokyo. What is the capital of Brazil? The capital of Brazil is Bras` | связно, эхо промпта |
| int4 / int4 | то же, что V int4 | — |
| f32 / **int8** | побайтово тот же текст, что f32 | V int8 безопасен |
| int4 / int8 + int8-акт (полный DSP-набор, нативные веса) | ` Paris, and the population of Paris is approximately 2.2 million people. …` | связно |

**В CPU-пути int4-KV сам по себе жёсткого цикла не даёт** (в отличие от
per-row int4-весов). Оговорка: V здесь квантуется с масштабом **на вектор
токена** (лучший случай). В DSP масштаб V обязан быть постоянным по токенам
(иначе PV-смешивание математически неверно), поэтому реальный ущерб от V там
**не меньше**, а скорее больше измеренного. Утверждение «виноват V, а не веса»
для CPU-пути не подтверждается; для DSP его надо перепроверить с глобальным
(или per-channel) масштабом V, а не per-token.

## 4. Финалисты на 40 токенах и втором промпте

P1, 40 токенов:

* Q4_K: ` Paris, and the capital of Japan is Tokyo. The capital of Brazil is Brasília, and the capital of Australia is Canberra. The capital of the United States is Washington, D.C. The`
* блок 128: ` Paris, and it is located in the north of the country. Paris is known for its culture, fashion, and history. The city is divided into 20 arrondissements, and it is one`
* блок 256 + H32: ` Paris, and the capital of Japan is Tokyo. The capital of Brazil is Brasília. The capital of the United States is Washington, D.C. The capital of Canada is Ottawa. The capital of` — ближе всего к f32 по траектории
* asym блок 256: ` Paris, the capital of the country located in Western Europe, and the most populous city in France. Paris is divided into 20 arrondissements, each of which is a city and a department`
* per-row + H32: ` Paris. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of South Korea is Seoul. The capital of Australia is Canberra. The capital of India is New Delhi.`

P2, 30 токенов:

* Q4_K: ` Jupiter. It is the largest and most massive planet, and its mass is more than twice that of all the other planets combined. Jupiter has a diameter`
* блок 128: ` Jupiter, which is about 142,984 kilometers in diameter. Jupiter is a gas giant planet, meaning it is composed primarily of hydrogen and helium gases`
* asym блок 256: ` Jupiter, with a diameter of about 142,000 kilometers (about 8.9 times the Earth's diameter). Jupiter has 95 known moons`
* per-row + H32: ` Jupiter. Jupiter is a gas giant, meaning it is composed mostly of hydrogen and helium. Gas giants like Jupiter have strong winds …`
* **per-row int4:** ` Jupiter. It is the fourth largest planet and the most powerful one. It is the largest and most powerful planet in the Solar System. The largest and` — фактическая ошибка + повтор, но не жёсткий цикл
* блок 256: ` Jupiter, and its mass is about 1,348,000 times the mass of the Earth. The largest planet in the Solar System is Jupiter,` — связно, эхо промпта

То есть per-row int4 деградирует на обоих промптах, но форма деградации
зависит от промпта (на P1 — цикл служебных слов, на P2 — повтор и фактическая
ошибка). Это ещё один аргумент за метрику на наборе промптов, а не за один
greedy-прогон.

## 5. Отрицательные результаты, которые надо зафиксировать

* **Q4_K как «верхняя граница» ничего не проверяет:** чекпойнт уже Q4_K, поэтому
  `Q4K=1` — это ровно f32-путь. Он воспроизводит эталонный текст, и это
  доказывает, что CPU-арифметика не виновата.
* **Hadamard — не серебряная пуля.** H32 при per-row работает, H32 при
  блоках 128/32 — цикл; H128 при per-row — жёсткий цикл. Направление эффекта
  зависит от связки «ротация × масштаб», а не от одной ротации.
* **CONV16 (int16 для shortconv) не монотонно улучшает:** на блоке 256 + H32
  текст хороший, на per-row + H32 — цикл, хотя cos выше. На реальном DSP
  `CONV16` — другая операция (Y16 на выходе свёртки), поэтому CPU-аналог
  (int16 на весах) нельзя считать эквивалентом; переносить надо сам Y16.
* **cos(W) не предсказывает текст.** Примеры: блок 128+H32 (cos 0.99320) —
  цикл, per-row+H32 (cos 0.98824) — связно; CONV16 повышает cos и ломает
  текст. Для приёмки формата нужен end-to-end eval.
* **Сравнение идёт с Q4_K-деквантованными весами, а не с истинными f32.**
  Абсолютная ошибка относительно настоящей f32-модели больше; относительные
  сравнения конфигураций корректны, абсолютные «сколько до f32» — нет.

## 6. Рекомендация для прод-формата

**Минимальное изменение: заменить per-row масштаб int4 на per-256 (симметричный
absmax), без min-терма и без ротации.**

* Байты: 0.5020 → **0.5156** Б/вес (+2.7 % к весам, всё ещё на 8 % меньше
  Q4_K 0.5625).
* Качество: relRMSE 0.1843 → 0.1390; текст связный на обоих промптах, в
  DSP-комбинации (int8-акт + K int4 + V int8) — тоже.
* Блок 128 (0.5312 Б/вес, relRMSE 0.1263) — запас качества, если 256 окажется
  на границе на реальном eval. Блок ≤512; 1024 и per-row — запрещены.
* Порог по relRMSE для быстрого CI-гейта: **≤ 0.14**. Выше 0.16 — ждать
  вырождения.

Если бюджет байтов жёстко per-row (0.502) — **per-row + H32** (0.5020 Б/вес,
relRMSE 0.1547): текст связный и на P1, и на P2, и в DSP-комбинации. Цена —
ротация весов (оффлайн, бесплатно по трафику) и активаций (O(K log 32) на
matvec), плюс хрупкость: H32 при блоках 128/32 и H128 при per-row дают циклы,
поэтому обязателен eval-набор.

Min-терм (`asym`) — опционально: он даёт лучший relRMSE (0.1162 при блоке 256)
и отличные тексты, но требует +4 Б на блок и отдельного слагаемого в ядре.
Симметричного блока 256 достаточно, поэтому min-терм — второй шаг, а не
условие исправления.

## 7. Что переносить в DSP

1. **Per-256 (лучше 128) масштаб для int4-GEMV весов.** В `gemv`-ядре это
   значит: строку делить на 8 (16) сегментов по 256, каждый сегмент считать
   своим dot, умножать на свой масштаб и складывать. На HMX это ложится в
   существующие per-32 scale/bias-блоки с групповым накоплением. Формат
   блоба: таблица масштабов на 256 + nibble-коды, без изменения раскладки
   кодов.
2. **Если менять ядро дорого — H32-ротация весов и активаций до квантования.**
   Нулевой оверхед по трафику, но: (а) ротировать надо **до** int8-квантования
   активаций (порядок «rotate → quantize», иначе шум int8 растянется
   ротацией); (б) на CPU H32+per-row сработал, а H128+per-row и H32+блок 128/32
   — нет; (в) проверить на eval-наборе, а не на одном промпте.
3. **V-кэш не опускать ниже int8.** K int4 (per-head, per-token, как `kvpage`)
   в изоляции приемлем; V int4 с per-token масштабом на CPU дал лишь сдвиг
   темы, но в DSP масштаб V постоянен по токенам — этот режим надо измерить
   отдельно и, скорее всего, оставить V int8.
4. **Голову/эмбеддинги можно оставить в том же блочном int4.** f32-голова не
   требуется: per-row по телу ломается и с f32-головой, а блок 256 по телу с
   f32-головой — связный.
5. **`CONV16` переносить как есть (Y16 на выходе свёртки), а не как int16 на
   весах.** CPU-аналог на весах дал немонотонный результат; на блоке 256 + H32
   он текст не портил, но это не доказательство для DSP-операции.

## 8. Риски и ограничения

* **Хрупкость greedy-траекторий.** Один промпт, 24–40 токенов: конфигурации
  скачут между «связно», «другая тема» и «цикл» от возмущений в один LSB
  (пример: CONV16). Нужен eval на ≥20 промптах + perplexity/accuracy, иначе
  легко принять шум за формат.
* **cos(W) — не критерий.** Использовать relRMSE (порог ~0.14) плюс end-to-end
  текст; хранить оба в отчётах паковщика.
* **Источник уже квантован (Q4_K).** Ошибки измерены относительно Q4_K, а не
  f32; на F16-источнике абсолютные потери будут больше.
* **Округление.** Существующий CPU-контроль использует `round()`
  (half-away-from-zero), `myelin-pack` — `round_ties_even`. Разница суб-LSB,
  но при переносе фиксировать одно правило в паковщике и ядре.
* **Стоимость.** Per-256 масштаб — +2.7 % весов и +1 умножение/сегмент в
  GEMV; H32 — O(K log 32) на matvec и отдельный буфер под ротированный вход.
  На телефоне это надо мерить по времени, здесь не мерилось.
* **KV-вывод ограничен CPU-моделью V-масштаба** (per-token = лучший случай).
  Реальный DSP с фиксированным по токенам масштабом V может деградировать
  сильнее; это отдельный замер на устройстве.

## 9. Воспроизведение

```bash
cd myelin
cargo build --release -p myelin-core --features hf-tokenizer --example host_generate

G=./target/release/examples/host_generate
P="The capital of France is"

# эталон и дефект
MYELIN_CPU_Q4K=1          $G --prompt "$P" --max-tokens 40
MYELIN_CPU_INT4=1         $G --prompt "$P" --max-tokens 40

# матрица блоков / ротаций / min-терма
MYELIN_CPU_WBLOCK=256     $G --prompt "$P" --max-tokens 40
MYELIN_CPU_WBLOCK=128     $G --prompt "$P" --max-tokens 40
MYELIN_CPU_INT4=1 MYELIN_CPU_WHAD=32 $G --prompt "$P" --max-tokens 40
MYELIN_CPU_WASYM=1 MYELIN_CPU_WBLOCK=256 $G --prompt "$P" --max-tokens 40

# изоляция KV при нативных весах
MYELIN_CPU_Q4K=1 MYELIN_CPU_KVV=int4 $G --prompt "$P" --max-tokens 24
MYELIN_CPU_Q4K=1 MYELIN_CPU_KVK=int4 $G --prompt "$P" --max-tokens 24

# DSP-комбинация
MYELIN_CPU_WBLOCK=256 MYELIN_CPU_INT8_ACT=1 \
  MYELIN_CPU_KVK=int4 MYELIN_CPU_KVV=int8 $G --prompt "$P" --max-tokens 40
```

`wstats` печатается в stderr при любом активном ключе; для нативного формата —
с `MYELIN_CPU_WSTATS=1`.
