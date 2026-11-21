//! GPU-регрессия staging-кольца kvarn: батч, начинающийся не на границе
//! страницы, не должен коммитить чужие K/V.
//!
//! Тест игнорируется по умолчанию: нужен OpenCL-девайс и модель.
//!
//! ```text
//! LFM_KV=kvarn LFM_TEST_MODEL=models/LFM3-1.7B-Q4_K_M.gguf \
//!   cargo test -p lfm-gpu --release --test kvarn_chunks -- --ignored --nocapture
//! ```

use anyhow::Result;
use lfm_gpu::GpuModel;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut ab = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
    }
    ab / (aa.sqrt() * bb.sqrt() + 1e-12)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Эталон — prefill страница за страницей (начала чанков выровнены по 32).
/// Вариант — тот же батчевый путь, но один чанк начинается с `start`, а
/// `0..start` идёт отдельным (выровненным) батчем. Старая схема
/// «store всего чанка, потом commit» на невыровненном `start` затирает начало
/// страницы на стыке и коммитит чужие K/V.
#[test]
#[ignore = "нужен OpenCL-девайс и модель (LFM_TEST_MODEL); запускать с LFM_KV=kvarn"]
fn kvarn_unaligned_chunks_match_sequential() -> Result<()> {
    let model = std::env::var("LFM_TEST_MODEL")
        .unwrap_or_else(|_| "models/LFM3-1.7B-Q4_K_M.gguf".to_string());
    let ids: Vec<u32> = (0..256)
        .map(|i| ((i * 2654435761usize) % 60000) as u32 + 100)
        .collect();
    let mut gpu = GpuModel::load(std::path::Path::new(&model), 2048)?;
    let batch = gpu.prefill_batch_size();

    for start in [1usize, 31, 32, 33, 63] {
        let end = start + batch;

        // Эталон: чанки по странице (32 токена), начала выровнены.
        gpu.reset()?;
        let mut pos = 0;
        while pos < end {
            let next = (pos + 32).min(end);
            gpu.forward_batch(&ids[pos..next], pos, next == end)?;
            pos = next;
        }
        let want = gpu.logits()?;

        // Вариант: выровненный префикс и один длинный чанк с начала `start`.
        gpu.reset()?;
        let mut pos = 0;
        let step = batch.min(32);
        while pos < start {
            let next = (pos + step).min(start);
            gpu.forward_batch(&ids[pos..next], pos, false)?;
            pos = next;
        }
        gpu.forward_batch(&ids[start..end], start, true)?;
        let got = gpu.logits()?;

        let cos = cosine(&want, &got);
        println!(
            "start={start:<3} ntok={:<3} cos={cos:.6} max|dlogit|={:.4}",
            end - start,
            max_abs_diff(&want, &got)
        );
        // Порог отделяет порчу страницы от квантового шума kvarn: на эталоне
        // (тайлы по `kvarn_tile_rows`) cos >= 0.976, на старой схеме «store
        // всего чанка, потом commit» — 0.43…0.81 для start = 1, 31, 63.
        assert!(
            cos > 0.95,
            "start={start}: cos {cos:.6} — kvarn закоммитил чужую страницу"
        );
    }
    Ok(())
}
