use rayon::prelude::*;

pub const Q4K_BLOCK: usize = 144;
pub const Q4K_PACKED: usize = 148;
pub const Q4K_SCALES: usize = 4;
pub const Q4K_QUANTS: usize = 20;

fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    match threads {
        0 | 1 => f(),
        n => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(pool) => pool.install(f),
            Err(_) => f(),
        },
    }
}

struct RowSink {
    q: *mut i8,
    scales: *mut f32,
    rows: usize,
}

unsafe impl Send for RowSink {}
unsafe impl Sync for RowSink {}

impl RowSink {
    fn new(q: &mut [i8], scales: &mut [f32], rows: usize) -> Self {
        Self {
            q: q.as_mut_ptr(),
            scales: scales.as_mut_ptr(),
            rows,
        }
    }

    unsafe fn store(&self, g: usize, r: usize, quant: &[i8; 32], scale: f32) {
        unsafe {
            *self.scales.add(g * self.rows + r) = scale;
            std::ptr::copy_nonoverlapping(
                quant.as_ptr(),
                self.q.add((g * self.rows + r) * 32),
                quant.len(),
            );
        }
    }
}

fn q4_i8_row(src: &[u8], per_row: usize, cols: usize, r: usize, sink: &RowSink) {
    let row = &src[r * per_row..(r + 1) * per_row];
    let mut vals = [0f32; 32];
    let mut quant = [0i8; 32];
    for b in 0..cols / 256 {
        let blk = &row[b * Q4K_PACKED..(b + 1) * Q4K_PACKED];
        let d = lfm_quant::half_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = lfm_quant::half_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        for jj in 0..8usize {
            let sc = blk[Q4K_SCALES + 2 * jj] as f32;
            let mn = blk[Q4K_SCALES + 2 * jj + 1] as f32;
            let qq = jj / 2;
            let hi = jj % 2 == 1;
            for i in 0..32usize {
                let byte = blk[Q4K_QUANTS + qq * 32 + i];
                let nib = if hi { byte >> 4 } else { byte & 0x0F } as f32;
                vals[i] = d * sc * nib - dmin * mn;
            }
            let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for i in 0..32 {
                quant[i] = (vals[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
            let g = b * 8 + jj;
            let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            unsafe { sink.store(g, r, &quant, scale) };
        }
    }
}

fn q6_i8_row(
    src: &[u8],
    block_bytes: usize,
    cols: usize,
    r: usize,
    sink: &RowSink,
) -> anyhow::Result<()> {
    let row = &src[r * block_bytes..(r + 1) * block_bytes];
    let vals = lfm_quant::dequant_row_q6_k(row, cols)?;
    let mut quant = [0i8; 32];
    for g in 0..cols / 32 {
        let chunk = &vals[g * 32..(g + 1) * 32];
        let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for i in 0..32 {
            quant[i] = (chunk[i] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        unsafe { sink.store(g, r, &quant, scale) };
    }
    Ok(())
}

pub fn prepack_q4_k(src: &[u8]) -> anyhow::Result<Vec<u8>> {
    if src.is_empty() || !src.len().is_multiple_of(Q4K_BLOCK) {
        anyhow::bail!(
            "q4_k bytes {} not a non-zero multiple of {Q4K_BLOCK}",
            src.len()
        );
    }
    let blocks = src.len() / Q4K_BLOCK;
    let mut out = vec![0u8; blocks * Q4K_PACKED];
    for b in 0..blocks {
        let s = &src[b * Q4K_BLOCK..(b + 1) * Q4K_BLOCK];
        let d = &mut out[b * Q4K_PACKED..(b + 1) * Q4K_PACKED];
        d[..4].copy_from_slice(&s[..4]);
        for j in 0..8 {
            let (sc, m) = scale_min(s, j);
            d[Q4K_SCALES + 2 * j] = sc;
            d[Q4K_SCALES + 2 * j + 1] = m;
        }
        d[Q4K_QUANTS..Q4K_QUANTS + 128].copy_from_slice(&s[16..144]);
    }
    Ok(out)
}

pub const Q6K_BLOCK: usize = 210;

pub fn prepack_q6_i8(src: &[u8], cols: usize) -> anyhow::Result<(Vec<i8>, Vec<f32>)> {
    prepack_q6_i8_with_threads(src, cols, 0)
}

pub fn prepack_q6_i8_with_threads(
    src: &[u8],
    cols: usize,
    threads: usize,
) -> anyhow::Result<(Vec<i8>, Vec<f32>)> {
    if cols == 0 || !cols.is_multiple_of(256) {
        anyhow::bail!("q6 cols {cols} not a non-zero multiple of 256");
    }
    let block_bytes = cols / 256 * Q6K_BLOCK;
    if src.is_empty() || !src.len().is_multiple_of(block_bytes) {
        anyhow::bail!(
            "q6 bytes {} not a non-zero multiple of {block_bytes}",
            src.len()
        );
    }
    let rows = src.len() / block_bytes;
    let groups = cols / 32;
    let mut q = vec![0i8; groups * rows * 32];
    let mut scales = vec![0f32; groups * rows];
    with_threads(threads, || {
        let sink = RowSink::new(&mut q, &mut scales, rows);
        if threads == 1 {
            for r in 0..rows {
                q6_i8_row(src, block_bytes, cols, r, &sink)?;
            }
            Ok(())
        } else {
            (0..rows)
                .into_par_iter()
                .try_for_each(|r| q6_i8_row(src, block_bytes, cols, r, &sink))
        }
    })?;
    Ok((q, scales))
}

pub fn scale_min(blk: &[u8], j: usize) -> (u8, u8) {
    let q = &blk[4..16];
    let jm = j & 3;
    let h = j >> 2;
    let a = q[jm];
    let b = q[jm + 4];
    let c = q[jm + 8];
    let lo_d = a & 63;
    let lo_m = b & 63;
    let hi_d = (c & 15) | ((a >> 6) << 4);
    let hi_m = (c >> 4) | ((b >> 6) << 4);
    if h == 0 { (lo_d, lo_m) } else { (hi_d, hi_m) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(seed: u8, d: u16, dmin: u16) -> Vec<u8> {
        let mut b = vec![0u8; Q4K_BLOCK];
        b[0..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&dmin.to_le_bytes());
        for i in 4..16 {
            b[i] = seed.wrapping_mul(31).wrapping_add(i as u8 * 7);
        }
        for i in 16..144 {
            b[i] = seed.wrapping_add((i as u8).wrapping_mul(13));
        }
        b
    }

    #[test]
    fn packed_size_and_layout() {
        let src = block(1, 0x3c00, 0x3400);
        let out = prepack_q4_k(&src).unwrap();
        assert_eq!(out.len(), Q4K_PACKED);
        assert_eq!(&out[..4], &src[..4]);
        assert_eq!(&out[Q4K_QUANTS..Q4K_QUANTS + 128], &src[16..144]);
    }

    #[test]
    fn scales_match_source_layout() {
        let src = block(9, 0x3c00, 0x3400);
        let out = prepack_q4_k(&src).unwrap();
        for j in 0..8 {
            let (sc, m) = scale_min(&src, j);
            assert_eq!(out[Q4K_SCALES + 2 * j], sc, "scale {j}");
            assert_eq!(out[Q4K_SCALES + 2 * j + 1], m, "min {j}");
        }
    }

    #[test]
    fn kernel_lane_window_is_four_bytes() {
        let src = block(5, 0x3c00, 0x3400);
        let out = prepack_q4_k(&src).unwrap();
        for u in 0..8usize {
            let sb = (u >> 1) * 2;
            let off = Q4K_SCALES + 4 * (u >> 1);
            let (sc0, m0) = scale_min(&src, sb);
            let (sc1, m1) = scale_min(&src, sb + 1);
            assert_eq!(out[off], sc0);
            assert_eq!(out[off + 1], m0);
            assert_eq!(out[off + 2], sc1);
            assert_eq!(out[off + 3], m1);
        }
    }

    #[test]
    fn multiple_blocks_are_independent() {
        let mut src = block(3, 0x3c00, 0x3400);
        src.extend_from_slice(&block(200, 0x3800, 0x3000));
        let out = prepack_q4_k(&src).unwrap();
        assert_eq!(out.len(), 2 * Q4K_PACKED);
        for j in 0..8 {
            let (sc, m) = scale_min(&src, j);
            assert_eq!(out[Q4K_SCALES + 2 * j], sc);
            assert_eq!(out[Q4K_SCALES + 2 * j + 1], m);
            let base = Q4K_PACKED + Q4K_SCALES + 2 * j;
            let (sc2, m2) = scale_min(&src[Q4K_BLOCK..], j);
            assert_eq!(out[base], sc2);
            assert_eq!(out[base + 1], m2);
        }
        assert_eq!(
            &out[Q4K_PACKED + Q4K_QUANTS..Q4K_PACKED + Q4K_QUANTS + 128],
            &src[Q4K_BLOCK + 16..Q4K_BLOCK + 144]
        );
    }

    #[test]
    fn six_bit_fields_cover_all_patterns() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let mut src = vec![0u8; Q4K_BLOCK];
                src[4] = a;
                src[8] = b;
                src[5] = a ^ 0x5a;
                src[9] = b ^ 0x33;
                src[12] = 0b0000_1010;
                src[13] = 0b1111_0101;
                let out = prepack_q4_k(&src).unwrap();
                let (a2, b2) = (src[5], src[9]);
                let expect = [
                    (a & 63, b & 63),
                    (a2 & 63, b2 & 63),
                    (0, 0),
                    (0, 0),
                    (
                        (src[12] & 15) | ((a >> 6) << 4),
                        (src[12] >> 4) | ((b >> 6) << 4),
                    ),
                    (
                        (src[13] & 15) | ((a2 >> 6) << 4),
                        (src[13] >> 4) | ((b2 >> 6) << 4),
                    ),
                ];
                for (j, (sc, m)) in expect.iter().enumerate() {
                    if *sc == 0 && *m == 0 {
                        continue;
                    }
                    let got = scale_min(&src, j);
                    assert_eq!(got, (*sc, *m), "j={j} a={a} b={b}");
                    assert_eq!(out[Q4K_SCALES + 2 * j], *sc, "packed scale j={j}");
                    assert_eq!(out[Q4K_SCALES + 2 * j + 1], *m, "packed min j={j}");
                }
            }
        }
    }

    #[test]
    fn rejects_unaligned_input() {
        assert!(prepack_q4_k(&[0u8; 143]).is_err());
    }

    fn q6_block(seed: u8) -> Vec<u8> {
        let mut b = vec![0u8; Q6K_BLOCK];
        b[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        for i in 0..16 {
            b[192 + i] = (i as i8 - 8) as u8;
        }
        for i in 0..128 {
            b[i] = (i as u8).wrapping_mul(37).wrapping_add(seed);
        }
        for i in 0..64 {
            b[128 + i] = (i as u8).wrapping_mul(53).wrapping_add(seed);
        }
        b
    }

    #[test]
    fn q6_i8_layout_and_precision() {
        let cols = 512;
        let mut src = Vec::new();
        for seed in 0..2u8 {
            src.extend_from_slice(&q6_block(seed * 11));
            src.extend_from_slice(&q6_block(seed * 7 + 3));
        }
        let (q, sc) = prepack_q6_i8(&src, cols).unwrap();
        assert_eq!(q.len(), cols * 2);
        assert_eq!(sc.len(), (cols / 32) * 2);
        for r in 0..2 {
            let row = &src[r * (cols / 256) * Q6K_BLOCK..(r + 1) * (cols / 256) * Q6K_BLOCK];
            let vals = lfm_quant::dequant_row_q6_k(row, cols).unwrap();
            for g in 0..cols / 32 {
                let scale = sc[g * 2 + r];
                for i in 0..32 {
                    let deq = q[(g * 2 + r) * 32 + i] as f32 * scale;
                    let want = vals[g * 32 + i];
                    assert!(
                        (deq - want).abs() <= scale * 0.5001 + 1e-4,
                        "r {r} g {g} i {i}: {deq} vs {want} scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn q6_i8_rejects_unaligned_cols() {
        assert!(prepack_q6_i8(&[0u8; 210], 128).is_err());
    }
    fn q4_block(seed: u8) -> Vec<u8> {
        let mut b = vec![0u8; Q4K_BLOCK];
        b[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x1c00u16.to_le_bytes());
        for i in 0..12 {
            b[4 + i] = (i as u8).wrapping_mul(17).wrapping_add(seed);
        }
        for i in 0..128 {
            b[16 + i] = (i as u8)
                .wrapping_mul(43)
                .wrapping_add(seed.wrapping_mul(7));
        }
        b
    }

    #[test]
    fn q4_i8_matches_reference_dequant() {
        let cols = 256usize;
        let rows = 2usize;
        let mut raw = Vec::new();
        for r in 0..rows {
            raw.extend_from_slice(&q4_block(r as u8 * 13));
        }
        let packed = prepack_q4_k(&raw).unwrap();
        let (q, sc) = prepack_q4_i8(&packed, cols).unwrap();
        let groups = cols / 32;
        assert_eq!(q.len(), rows * cols);
        assert_eq!(sc.len(), groups * rows);
        for r in 0..rows {
            let row = &raw[r * Q4K_BLOCK..(r + 1) * Q4K_BLOCK];
            let reference = lfm_quant::dequant_row_q4_k(row, cols).unwrap();
            for g in 0..groups {
                let scale = sc[g * rows + r];
                assert!(scale.is_finite() && scale >= 0.0);
                for i in 0..32 {
                    let got = q[(g * rows + r) * 32 + i] as f32 * scale;
                    let want = reference[g * 32 + i];
                    assert!(
                        (got - want).abs() <= scale.max(1e-6),
                        "row {r} group {g} i {i}: {got} vs {want} (scale {scale})"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_i8_rejects_bad_shapes() {
        assert!(prepack_q4_i8(&[], 256).is_err());
        assert!(prepack_q4_i8(&[0u8; Q4K_PACKED], 0).is_err());
        assert!(prepack_q4_i8(&[0u8; Q4K_PACKED], 512).is_err());
    }

    #[test]
    fn q4_i8_parallel_matches_sequential() {
        let cols = 512usize;
        let rows = 5usize;
        let mut raw = Vec::new();
        for r in 0..rows {
            for b in 0..cols / 256 {
                raw.extend_from_slice(&q4_block((r * 3 + b) as u8 * 11 + 1));
            }
        }
        let packed = prepack_q4_k(&raw).unwrap();
        let (q_ref, sc_ref) = prepack_q4_i8_with_threads(&packed, cols, 1).unwrap();
        for threads in [0usize, 2, 3, 8] {
            let (q, sc) = prepack_q4_i8_with_threads(&packed, cols, threads).unwrap();
            assert_eq!(q, q_ref, "threads={threads}");
            assert_eq!(sc, sc_ref, "threads={threads}");
        }
        let (q, sc) = prepack_q4_i8(&packed, cols).unwrap();
        assert_eq!(q, q_ref);
        assert_eq!(sc, sc_ref);
    }

    #[test]
    fn q6_i8_parallel_matches_sequential() {
        let cols = 512usize;
        let rows = 4usize;
        let mut raw = Vec::new();
        for r in 0..rows {
            for b in 0..cols / 256 {
                raw.extend_from_slice(&q6_block((r * 5 + b) as u8 * 13 + 2));
            }
        }
        let (q_ref, sc_ref) = prepack_q6_i8_with_threads(&raw, cols, 1).unwrap();
        for threads in [0usize, 2, 3, 8] {
            let (q, sc) = prepack_q6_i8_with_threads(&raw, cols, threads).unwrap();
            assert_eq!(q, q_ref, "threads={threads}");
            assert_eq!(sc, sc_ref, "threads={threads}");
        }
        let (q, sc) = prepack_q6_i8(&raw, cols).unwrap();
        assert_eq!(q, q_ref);
        assert_eq!(sc, sc_ref);
    }
}

pub fn prepack_q4_i8(src: &[u8], cols: usize) -> anyhow::Result<(Vec<i8>, Vec<f32>)> {
    prepack_q4_i8_with_threads(src, cols, 0)
}

pub fn prepack_q4_i8_with_threads(
    src: &[u8],
    cols: usize,
    threads: usize,
) -> anyhow::Result<(Vec<i8>, Vec<f32>)> {
    if cols == 0 || !cols.is_multiple_of(256) {
        anyhow::bail!("q4 cols {cols} not a non-zero multiple of 256");
    }
    let per_row = cols / 256 * Q4K_PACKED;
    if src.is_empty() || !src.len().is_multiple_of(per_row) {
        anyhow::bail!(
            "q4 packed bytes {} not a non-zero multiple of {per_row}",
            src.len()
        );
    }
    let rows = src.len() / per_row;
    let groups = cols / 32;
    let mut q = vec![0i8; groups * rows * 32];
    let mut scales = vec![0f32; groups * rows];
    with_threads(threads, || {
        let sink = RowSink::new(&mut q, &mut scales, rows);
        if threads == 1 {
            for r in 0..rows {
                q4_i8_row(src, per_row, cols, r, &sink);
            }
        } else {
            (0..rows)
                .into_par_iter()
                .for_each(|r| q4_i8_row(src, per_row, cols, r, &sink));
        }
    });
    Ok((q, scales))
}
