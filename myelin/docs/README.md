# Документация Myelin

Myelin — мобильный LLM-рантайм для Hexagon NPU (LFM3, Snapdragon 888 / CDSP v68
как первый стенд). Документация описывает то, что реально сделано и измерено,
и отдельно помечает то, что ещё в работе.

## Индекс

| Файл | О чём |
|---|---|
| [SETUP.md](SETUP.md) | **Установка с нуля**: что скачать (SDK 6.7, NDK, QEMU, модель), переменные, сборка, стенды, телефон, частые ошибки |
| [HANDOVER.md](HANDOVER.md) | **Передача проекта**: что работает, решения (Q4_K — продуктовый формат), цифры, открытые задачи, порядок работ |
| [arch-v2.md](arch-v2.md) | План v2: форматы, батчинг, мульти-PD, гейты, владение файлами |
| [kernel-contract.md](kernel-contract.md) | Контракт ядер из OpenCL-эталона: Q4_K, int8/32, GEMV/FFN/attention |
| [kv-batch-contract.md](kv-batch-contract.md) | Paged KV, kvarn, радикс, continuous batching; разбор KV-бага |
| [multipd-contract.md](multipd-contract.md) | Раскладка по PD, редукция, протокол, лимиты |
| [decode-plan.md](decode-plan.md) | Разложение 109 мс/токен и приоритеты O1–O6 |
| [transport-results.md](transport-results.md) | Что дали O1/O2/O4 на практике (O1 не подтвердился) |
| [quality-int4-cpu.md](quality-int4-cpu.md) | Почему per-row масштаб ломает текст и что чинит |
| [attn-fused-contract.md](attn-fused-contract.md) | Сведение attention 3 RPC → 1: команды, поля, семантика |
| [architecture.md](architecture.md) | Слои (Kotlin → UniFFI → Rust → FFI → DSP), крейты, поток одного шага, разделение host/DSP, `ModelSpec`/`Arch`, как добавить модель |
| [dsp-kernels.md](dsp-kernels.md) | Наши DSP-ядра: int4-GEMV, FFN одним RPC, paged attention, shortconv, KV store/commit, SiLU; цифры и условия замеров |
| [paged-kv.md](paged-kv.md) | Страницы по 32 токена, таблица страниц, radix-кэш на хосте, раскладка kvarn-страницы, staging-кольцо, инварианты и краевые случаи |
| [hvx.md](hvx.md) | Внутренности HVX: контексты, потоки и стеки, `vrmpy`/`vsum`/`vshuff`, `l2fetch`, exp, грабли v68 |
| [hmx.md](hmx.md) | Внутренности HMX: питание, раскладки fp16/int8/int4, `:deep`, bias/scale, MAC-геометрия, грабли v68; работает, prefill упирается в копирование весов |
| [qnn-notes.md](qnn-notes.md) | Что выяснено про QNN: сборка графа, ограничения V68, context binary, скорости, дизассемблер скела |
| [measurements.md](measurements.md) | Сводная таблица замеров: APU для сравнения, телефон HVX/HMX/QNN, с датами и условиями |
| [decisions.md](decisions.md) | ADR: почему «всё сами», paged KV, kvarn 4/2, `kv_pool = max_seqs`, HMX в prefill (ADR-010), UniFFI, in-app, SDK 37 |
| [roadmap.md](roadmap.md) | Этапы P0…оптимизации с критериями приёмки и текущим статусом |
| [bibliography.md](bibliography.md) | Источники (PRM, QNN, llama.cpp, htp-ops-lib, Android) и что из них взято |

## Обозначения

* **измерено** — есть вывод раннера/отчёт с условиями (устройство, дата, источник).
* **оценка** — расчёт или экстраполяция, не проверенная на стенде.
* **в работе** — код/эксперимент начат, результат не закрыт.
* **нет данных** — в доступных источниках цифры/факта нет; ничего не выдумываем.

Все замеры телефона сняты на Snapdragon 888 (CDSP v68, unsigned PD), если не
указано иное; клок CDSP измерен 1132–1421 pcycles/µs (1.13–1.42 ГГц, DVFS
гуляет), даты — по времени артефактов в `/tmp/opencode`.

## Статус одной строкой

Каркас рантайма (Rust + C ABI + UniFFI + APK-демо) собран и работает на
заглушке; DSP-ядра (int4 GEMV, FFN, attention, shortconv) написаны и измерены
отдельно от рантайма; FFN и shortconv прошли первые прогоны на устройстве.
FastRPC-клиент в `myelin-npu` написан (с двумя багами, REPORT20 §3.2), но FFI
пока использует mock.
