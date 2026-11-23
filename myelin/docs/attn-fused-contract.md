# Контракт `KVSTORE` и `ATTN_SCALE`: attention 2 RPC → 1

Документ фиксирует DSP-часть O4 из `decode-plan.md` §5/O4. Хостовая часть
уже реализована в `myelin-core` (`DspBackend::attention_fused`) и включается
переменной `MYELIN_ATTN_FUSED=1` с автоматическим откатом на старый путь,
если скелет отвергает список. **`lfmgemv_impl.c` этим изменением не
трогался**: ниже ровно то, что нужно добавить в `lfm_run_cmd` и в
`lfm_cmd`-комментарий.

## 1. Зачем

Сейчас attention-слой — 3 `exec` + 1 `kv_write`:

```
exec qkv        (q|k|v|gate в out)
host: dequant, per-head RMSNorm, rope, int8 q, int4/int8 K/V
kv_write        (K/KT/V в резидентные образы)
exec ATTN       (q12 в out)
host: q12 * v_scale[dim] * gate[head] -> int8
exec o_proj
```

`KVSTORE` убирает `kv_write`, `ATTN_SCALE` убирает хост-цикл и позволяет
положить `ATTN + ATTN_SCALE + o_proj` в один список. Остаётся 2 `exec` на
слой: `qkv` (до него нужен rope на хосте) и fused-список. 82 → 60 RPC на
токен (11 слоёв × 2 вместо 11 × 4 с `kv_write`), при 0.7 мс/вызов это
≈15 мс/токен на текущем телефоне.

Один `exec` на слой потребует ещё `NORMQ`/`ROPE` на DSP (O5) — это вне
этого контракта.

## 2. Общие правила

* Новые **типы команд** списка `exec`, не новые методы IDL: `lfmgemv_exec`
  уже принимает произвольный список, `LFM_CMD_MAX` не меняется.
* Номера: `LFM_CMD_KVSTORE = 20`, `LFM_CMD_ATTN_SCALE = 21`. Девятка уже
  занята `LFM_CMD_ROTQ` в параллельной работе по O5, поэтому берём
  свободный диапазон. Порядок и номера совпадают в
  `myelin-core/backend/dsp.rs` и `myelin-npu/method.rs`; тест
  `core_encoder_matches_the_dsp_layout` сверяет байты. В `lfm_run_cmd`
  нужно поднять и проверку `if (c->type > LFM_CMD_GEMV16)`, иначе новые
  команды отвергаются до `switch`.
* `lfm_cmd` не меняется по размеру (96 Б) — используются существующие поля.
* Оба типа работают только с `src = LFM_SRC_SCRATCH`. `dst` у `KVSTORE`
  не используется; у `ATTN_SCALE` — `LFM_DST_SCRATCH`.
* Ошибки — `AEE_EBADPARM` с индексом команды в `times[1]`, как у остальных.

## 3. `KVSTORE` (9)

Переносит уже посчитанные хостом K/KT/V одного токена в резидентные образы.
Побайтовая замена `lfmgemv_kv_write` (тот же layout, тот же `memcpy`).

Поля команды:

| поле | значение |
|---|---|
| `type` | 20 |
| `src` | `LFM_SRC_SCRATCH` |
| `h0` | `hk` — int4-кэш K (`4 * slots * 64` Б, token-major) |
| `h1` | `hkt` — per-token Q16-масштабы (`4 * slots * 4` Б) |
| `h2` | `hv` — int8-кэш V (`4 * slots * 128` Б) |
| `src_off` | начало блока `K|KT|V` в scratch |
| `src_len` | `k_bytes + kt_bytes + v_bytes` |
| `a0` | `slots` — всего слотов (`cache_pages * 32`) |
| `a1` | `slot` — если `rt_base < 0`; иначе значение берётся из `rt[rt_base]` |
| `a2` | `k_bytes` (`heads * 64`) |
| `a3` | `kt_bytes` (`heads * 4`) |
| `rt_base` | `>= 0`: `slot = rt[rt_base]` (хост передаёт `rt[4] = slot`) |

`v_bytes = src_len - a2 - a3`, `heads = k_bytes / 64`, и должно быть
`heads == kt_bytes / 4 == v_bytes / 128`.

Семантика (как в `lfmgemv_kv_write`):

```c
for (i = 0; i < heads; i++) {
    size_t row = (size_t)i * (size_t)slots + (size_t)slot;
    size_t ko = row * 128, to = row * 4, vo = row * 128;
    memcpy(bk->w + ko,         k  + i * 64,  64);
    memset(bk->w + ko + 64, 0, 64);            /* token-major: верхние 64 Б */
    memcpy(bt->w + to,         kt + i * 4,    4);
    memcpy(bv->w + vo,         v  + i * 128, 128);
}
```

Проверки: `h0/h1/h2` в `0..MAXBUF`, буферы `used && w`, `slots >= 1`,
`0 <= slot < slots`, длины кратны 64/4/128 и сходятся по `heads`,
`lfm_range_ok(src_off, src_len, g_scratch_bytes)`, `ko+128 <= bk->wbytes` и
т.д. `times` не заполняется.

Хост (`attention_layer_full`) кладёт блок сразу за `q@0`: `src_off = hidden`,
`src_len = 256 + 16 + 512 = 784` для 4 kv-голов, `slot = harness`.

## 4. `ATTN_SCALE` (10)

`real[i] = q12[i] * attn_out_scale * vscale[kh*128 + dim] * gate[qh]`,
`dst[i] = clamp(round(real[i] * o_act_scale), -127, 127)`.

Порядок умножений фиксирован: `((q12 * attn_out_scale) * vscale) * gate`,
затем `* o_act_scale` — как в старом хост-цикле `dsp.rs:1766-1786`.

Поля:

| поле | значение |
|---|---|
| `type` | 21 |
| `src` | `LFM_SRC_SCRATCH` |
| `dst` | `LFM_DST_SCRATCH` |
| `h0` | handle таблицы `kv_heads*128` f32 (V scales), `-1` = без таблицы |
| `src_off` | int32 Q12, `16*128` значений (8192 Б) |
| `dst_off` | int8, `16*128` значений (2048 Б) |
| `tbl_off` | offset `heads` f32 гейта в scratch |
| `a0` | биты f32 `attn_out_scale` |
| `a1` | биты f32 `o_act_scale` |
| `a2` | длина гейта (`16`, либо `0` — гейта нет) |
| `a3` | 0 (резерв) |

`h0 == -1` означает `vscale[...] = 1.0` (глобальный V-масштаб хост уже
свернул в `a0`). `heads = 16`, `head_dim = 128`, `kv_heads = 4` — те же
константы `ATTN_HQ/ATTN_D/ATTN_KV`, что у `attn_core`.

```c
static int attn_scale_cmd(const lfm_cmd *c)
{
    const int32_t *src = (const int32_t *)(g_scratch + c->src_off);
    int8_t *dst = (int8_t *)(g_scratch + c->dst_off);
    const float *gate = (const float *)(g_scratch + c->tbl_off);
    const float *vscale = NULL;
    float m_out, m_act;
    int i, n = ATTN_HQ * ATTN_D;

    if (c->h0 >= 0) {
        gbuf_t *b = &g_buf[c->h0];
        if (!b->used || !b->w || b->wbytes < ATTN_KV * ATTN_D * 4) return AEE_EBADPARM;
        vscale = (const float *)b->w;
    }
    memcpy(&m_out, &c->a0, 4);
    memcpy(&m_act, &c->a1, 4);
    if (c->a2 > ATTN_HQ) return AEE_EBADPARM;
    if (!lfm_range_ok(c->src_off, n * 4, g_scratch_bytes) ||
        !lfm_range_ok(c->dst_off, n, g_scratch_bytes) ||
        (c->a2 > 0 && !lfm_range_ok(c->tbl_off, c->a2 * 4, g_scratch_bytes)))
        return AEE_EBADPARM;
    for (i = 0; i < n; i++) {
        int qh = i / ATTN_D, kh = qh / (ATTN_HQ / ATTN_KV), d = i % ATTN_D;
        float v = vscale ? vscale[kh * ATTN_D + d] : 1.0f;
        float real = (float)src[i] * m_out * v;
        int32_t code;
        if (c->a2 > 0) real *= gate[qh];
        real *= m_act;
        code = (int32_t)(real >= 0 ? real + 0.5f : real - 0.5f);
        if (code > 127) code = 127;
        if (code < -127) code = -127;
        dst[i] = (int8_t)code;
    }
    return 0;
}
```

Замечание по numerics: старый хост округляет `f32::round` (half away from
zero), `real` считается в f32. Команда должна повторять это; расхождение
допустимо только в пределах ±1 кода int8 и гейтится `cos ≥ 0.999` на
24 токенах.

## 5. Список и `act` хоста

`MYELIN_ATTN_FUSED=1` собирает на слой:

```
0  KVSTORE  (k@2048, kt@2304, v@2320; rt[4]=slot)
1  ATTN     (q@0, q12 -> scratch[4096], tbl в scratch[table_off],
             tbl_len=cache_pages, rt[0..3]=pos/seq/window/reps)
2  ATTN_SCALE (q12@4096 -> int8@16384, vscale=h0, gate@gate_off)
3  GEMV     (o_proj: int8@16384 -> OUT[0..hidden])
```

`act` (все смещения — байты в scratch, `exec` копирует `act` в 0):

| диапазон | что |
|---|---|
| `0 .. 2048` | int8 q |
| `2048 .. 2304` | K, 4 × 64 |
| `2304 .. 2320` | KT, 4 × 4 |
| `2320 .. 2832` | V, 4 × 128 |
| `2832 .. 2896` | gate f32, 16 значений (если есть) |
| `2896 .. 2896+4*pages` | identity block table, `pages = cache_pages` |

`rt = [pos, seq, window, reps, slot]`, `slot = table[pos/32]*32 + pos%32`
(для identity-таблицы — `pos`).

Скретч-смещения: `SCRATCH_ATTN_Q12 = 4096` (8192 Б), `SCRATCH_ATTN_I8 =
16384` (2048 Б). Конфликта с CONV-кольцами (с 1 МиБ) нет.

## 6. V-scale таблица

`model_pack::weight_plan` теперь кладёт `attention.v_channel` (512 f32)
четвёртым резидентым образом attention-слоя (`blk.N.vscale`), поэтому
`AttentionStage::value_scales` — валидный handle. Если handle `<= 0` или
`VQuant::Global`, хост не включает fused-путь и остаётся на трёх вызовах:
`attn_core` сам `hvs` не читает, так что старый путь не затронут.

## 7. Гейты

1. Юнит: `core_encoder_matches_the_dsp_layout` (байты), `attention_list_is_stable...`.
2. После реализации DSP: `MYELIN_ATTN_FUSED=1` на 24 токенах, ctx 256/512:
   те же токены, что без флага (`TOKENS` совпадает), `RESET_CHECK PASS`.
3. `MYELIN_TRACE=1`: в `exec transport` число вызовов на токен 82 → 60.
4. Если скелет вернул `AEE_EBADPARM`, `DspBackend` печатает
   `fused attention unsupported, falling back: …` и до конца сессии идёт
   старым путём (KVSTORE идемпотентен: повторная запись того же слота).
