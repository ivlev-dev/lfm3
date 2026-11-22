# hvx — HVX-ядра (Hexagon v68): GEMV, FFN, attention

Всё считается на CDSP в одном FastRPC-вызове на слой; веса живут на DSP
резидентно. Стенд не требует устройства для сборки, но все цифры — с телефона.

## Файлы

| файл | что |
|---|---|
| `lfmgemv.idl` | FastRPC-интерфейс: `gemv`, `ffn`, `attn`, `silu`, `probe`, `attach_out`, … |
| `lfmgemv_impl.c` | DSP-ядра: int4/int8 GEMV, SiLU-LUT, FFN, paged attention, PV |
| `host_main.c` | базовый тест GEMV (корректность + полоса) |
| `ab_main.c` | A/B варианты ядра, пробы полосы, multi-context |
| `ffn_main.c` | тест FFN-слоя (gate+up → SiLU → down), bit-exact проверка |
| `attn_main.c` | тест attention: CPU-f32 эталон, cos/max_abs_err, тайминги, проверка exp |
| `diag_main.c`, `probe_main.c` | мелкие диагностические прогоны |
| `silu_lut.h`, `exptab.h` | таблицы SiLU и exp |
| `silu_study.c`, `poly_fit.c`, `seg_fit.c`, `expcheck.c` | офлайн-числодробилки (SiLU/exp на хосте, без DSP) |
| `build.sh`, `run.sh` | сборка и пуш/запуск (`MADB`, `MDEV`) |

Не переносился `impl.s` (280 КБ дизассемблера, генерируется из `lfmgemv_impl.c`).

## Сборка и запуск

```bash
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.7.0.0 ./build.sh
export MADB=<adb-серийник>
./run.sh push
./run.sh lfmgemv_test                 # GEMV: корректность + полоса
./run.sh ab_test 3 3                  # A/B варианты: раунды, повторы
./run.sh ffn_test 2 2                 # FFN-слой, 2 контекста
./run.sh attn_test 64 0 2             # ctx 2048, полное окно, 2 контекста
./run.sh attn_test 64 256 2           # скользящее окно 256
./run.sh attn_test 64 0 2 pwl         # exp кусочно-линейный вместо bit-trick
```

`build.sh` собирает скелет (`-mv68 -mhvx -mhvx-length=128b`) и хостовые тесты
(NDK r27c, API 31). Тесты включают unsigned PD сами; скелет подхватывается через
`ADSP_LIBRARY_PATH` (его выставляет `run.sh`).

Офлайн-числодробилки собираются обычным `cc`, например:
`cc -O2 silu_study.c -lm -o silu_study`.

## Формат весов int4

Row-major `[N][K]`, `K/2` байт на строку, два signed int4 в байте:

```
byte = lo | (hi << 4)        lo -> чётный k, hi -> нечётный k
хранимый ниббл = w_signed + 8    (offset-8, 0..15)
```

Коррекция скалярная, раз на строку: `sum(w*a) = sum(u*a) - 8*sum(a)`.
Активации int8 `[M][K]`, деинтерлив в `ae`/`ao` (чётные/нечётные k). Выход int8
`[M][N]` с Q16-множителем на строку (`vmpyiewh` → clamp ±127) либо сырой int32.

## Ядро GEMV (M=1, K=2048)

На строку (1024 Б = 8 HVX-векторов): `vand 0x0f` → младшие нибблы, `vlsr 4` →
старшие (2 оп/вектор); `vrmpyacc` unsigned×signed по `ae`/`ao` (2 оп/вектор);
4 аккумулятора, дерево, `-8*sum(a)`; `l2fetch` на 8 КБ вперёд каждые 8 строк
(100% покрытие); опционально 2 или 4 строки за итерацию.

## Измерено (SD888, CDSP v68, unsigned PD)

| что | результат |
|---|---|
| int4 GEMV 7680×2048, 1 контекст | 0.71 мс, 11.0 GB/s (с `l2fetch`); без префетча 3.08 мс |
| int4 GEMV, 2/4 строки за итерацию | 0.63 мс, 12.5 GB/s, 25 GMAC/s |
| int8-веса (15.7 МБ) на том же стенде | 1.24–1.30 мс → 2.00× медленнее int4 |
| QNN int8 на том же тензоре | 1.37 мс (11.5 GB/s) |
| FastRPC round trip, тривиальное ядро | ~0.19–0.21 мс |
| **FFN-слой** (2×7680×2048 → SiLU → 2048×7680) | **1.45 мс**, bit-exact; исторически 3.11 мс на 1 контексте, ~1.98 мс на 2 |
| **attention, ctx 2048** | **1178 мкс** |
| multi-context 7680×2048 int4 | 1 ctx 608 мкс (12.9 GB/s), 2 ctx 347 мкс (22.7 GB/s), 4+ — упирается в 2 HVX-контекста |

## Что важно знать (иначе теряется день)

- **`qurt_thread_create` с HVX-функцией возвращает −1**, пока не задан явный
  адрес стека (`qurt_thread_attr_set_stack_addr`) — как в `libs/worker_pool`.
  Этот PD отдаёт 2 HVX 128B-контекста; `qurt_hvx_reserve` возвращает −1.
  Воркеры берут HVX-лок на задачу (держать контекст всю жизнь потока — дедлок).
- **`mulq16(a,b)` = `(a*b)>>16` в int32** — 64-битные хелперы на v68 дороги.
- **`pack_out_i8` скалярный**: `Q6_Vw_vmpyieacc_VwVwVuh` даёт неверные лейны.
- **SiLU**: LUT-1025 + lerp даёт max|err| 3e-5 (Q16); кусочно-квадратичная
  32 сегмента — 3e-4; единый полином deg 13/17 — 1.8e-2/4.4e-3 и переполняет
  16×16. В v68 нет HVX-float и gather, векторный SiLU — дерево констант-селектов.
- **Paged attention**: page = 32 токена, `slot = (tbl[pos>>5]<<5) | (pos&31)`,
  K int4 (64 Б/токен, offset-8, lo = чётная размерность), V int2
  (в тесте деквантуется хостом в int8: на DSP нет `Q6_Vh_vshuff_VhVh`).
  score в Q8, exp по умолчанию — bit-trick `exp2`, кусочно-линейный
  (`attn_test ... pwl`) даёт mean|err| 1.7e-2 против 8.1e-4 у bit-trick.
- **PV token-major**: V лежит транспонированно по 32-токенной странице
  (`byte 4*d'+j = V[32g+d'][32c+j]`), один `vrmpy` на (группу dim, 4 токена),
  естественный порядок без перестановки; нормализация — `2^28/L` один раз на
  голову, int64-умножение.
- **Резидентный выход**: `lfmgemv_attach_out(h, fd, nbytes)` мапит host
  rpcmem/ION fd через `apps_mem_share_map()`, `gemv_resident`/`ffn_resident`
  пишут сразу туда. Компилируется; на устройстве не замерялось.
- RPC-стоимость: ~0.15 мс базовая, +0.4–0.45 мс при возврате 7.7 КБ;
  rpcmem/ION для этого не помогли.
