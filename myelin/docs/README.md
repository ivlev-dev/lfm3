# Документация Myelin

Myelin — мобильный LLM-рантайм для Hexagon NPU (LFM3, Snapdragon 888 / CDSP v68
как первый стенд). Документация описывает то, что реально сделано и измерено,
и отдельно помечает то, что ещё в работе.

## Индекс

| Файл | О чём |
|---|---|
| [architecture.md](architecture.md) | Слои (Kotlin → UniFFI → Rust → FFI → DSP), крейты, поток одного шага, разделение host/DSP, `ModelSpec`/`Arch`, как добавить модель |
| [dsp-kernels.md](dsp-kernels.md) | Наши DSP-ядра: int4-GEMV, FFN одним RPC, paged attention, shortconv, KV store/commit, SiLU; цифры и условия замеров |
| [paged-kv.md](paged-kv.md) | Страницы по 32 токена, таблица страниц, radix-кэш на хосте, раскладка kvarn-страницы, staging-кольцо, инварианты и краевые случаи |
| [hvx.md](hvx.md) | Внутренности HVX: контексты, потоки и стеки, `vrmpy`/`vsum`/`vshuff`, `l2fetch`, exp, грабли v68 |
| [hmx.md](hmx.md) | Внутренности HMX: питание, 16-битные ячейки, раскладки, `:deep` на весах, MAC-геометрия, статус R&D |
| [qnn-notes.md](qnn-notes.md) | Что выяснено про QNN: сборка графа, ограничения V68, context binary, скорости, дизассемблер скела |
| [measurements.md](measurements.md) | Сводная таблица замеров: APU для сравнения, телефон HVX/HMX/QNN, с датами и условиями |
| [decisions.md](decisions.md) | ADR: почему «всё сами», paged KV, kvarn 4/2, `kv_pool = max_seqs`, HMX как R&D, UniFFI, in-app, SDK 37 |
| [roadmap.md](roadmap.md) | Этапы P0…оптимизации с критериями приёмки и текущим статусом |
| [bibliography.md](bibliography.md) | Источники (PRM, QNN, llama.cpp, htp-ops-lib, Android) и что из них взято |

## Обозначения

* **измерено** — есть вывод раннера/отчёт с условиями (устройство, дата, источник).
* **оценка** — расчёт или экстраполяция, не проверенная на стенде.
* **в работе** — код/эксперимент начат, результат не закрыт.
* **нет данных** — в доступных источниках цифры/факта нет; ничего не выдумываем.

Все замеры телефона сняты на Snapdragon 888 (CDSP v68, 1.42 ГГц, unsigned PD),
если не указано иное. Даты — по времени артефактов в `/tmp/opencode`.

## Статус одной строкой

Каркас рантайма (Rust + C ABI + UniFFI + APK-демо) собран и работает на
заглушке; DSP-ядра (int4 GEMV, FFN, attention, shortconv) написаны и измерены
отдельно от рантайма; транспорт FastRPC в `myelin-npu` ещё не подключён.
