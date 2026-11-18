pub const Q4K_BLOCK: usize = 144;
pub const Q4K_PACKED: usize = 148;
pub const Q4K_SCALES: usize = 4;
pub const Q4K_QUANTS: usize = 20;

pub fn prepack_q4_k(src: &[u8]) -> Vec<u8> {
    assert!(
        src.len() % Q4K_BLOCK == 0,
        "q4_k bytes {} not a multiple of {Q4K_BLOCK}",
        src.len()
    );
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
    out
}

pub const Q6K_BLOCK: usize = 210;

pub fn prepack_q6_i8(src: &[u8], cols: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(cols % 256 == 0, "q6 cols {cols} not a multiple of 256");
    let block_bytes = cols / 256 * Q6K_BLOCK;
    assert!(
        src.len() % block_bytes == 0,
        "q6 bytes {} not a multiple of {block_bytes}",
        src.len()
    );
    let rows = src.len() / block_bytes;
    let groups = cols / 32;
    let mut q = vec![0i8; groups * rows * 32];
    let mut scales = vec![0f32; groups * rows];
    for r in 0..rows {
        let row = &src[r * block_bytes..(r + 1) * block_bytes];
        let vals = lfm_quant::dequant_row_q6_k(row, cols).expect("q6 row dequant");
        for g in 0..groups {
            let chunk = &vals[g * 32..(g + 1) * 32];
            let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sc = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            scales[g * rows + r] = sc;
            let base = (g * rows + r) * 32;
            for i in 0..32 {
                q[base + i] = (chunk[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, scales)
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
    if h == 0 {
        (lo_d, lo_m)
    } else {
        (hi_d, hi_m)
    }
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
        let out = prepack_q4_k(&src);
        assert_eq!(out.len(), Q4K_PACKED);
        assert_eq!(&out[..4], &src[..4]);
        assert_eq!(&out[Q4K_QUANTS..Q4K_QUANTS + 128], &src[16..144]);
    }

    #[test]
    fn scales_match_source_layout() {
        let src = block(9, 0x3c00, 0x3400);
        let out = prepack_q4_k(&src);
        for j in 0..8 {
            let (sc, m) = scale_min(&src, j);
            assert_eq!(out[Q4K_SCALES + 2 * j], sc, "scale {j}");
            assert_eq!(out[Q4K_SCALES + 2 * j + 1], m, "min {j}");
        }
    }

    #[test]
    fn kernel_lane_window_is_four_bytes() {
        let src = block(5, 0x3c00, 0x3400);
        let out = prepack_q4_k(&src);
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
        let out = prepack_q4_k(&src);
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
                let out = prepack_q4_k(&src);
                let (a2, b2) = (src[5], src[9]);
                let expect = [
                    (a & 63, b & 63),
                    (a2 & 63, b2 & 63),
                    (0, 0),
                    (0, 0),
                    ((src[12] & 15) | ((a >> 6) << 4), (src[12] >> 4) | ((b >> 6) << 4)),
                    ((src[13] & 15) | ((a2 >> 6) << 4), (src[13] >> 4) | ((b2 >> 6) << 4)),
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
        assert!(std::panic::catch_unwind(|| prepack_q4_k(&[0u8; 143])).is_err());
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
        let (q, sc) = prepack_q6_i8(&src, cols);
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
        assert!(std::panic::catch_unwind(|| prepack_q6_i8(&[0u8; 210], 128)).is_err());
    }
}
