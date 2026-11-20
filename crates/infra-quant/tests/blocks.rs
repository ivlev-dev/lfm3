use lfm_quant::{
    dequant_q4_k, dequant_q6_k, dequant_q8_0, dequant_row_q4_k, dequant_row_q6_k, dot_row_q4_k,
    dot_row_q6_k, f16s_to_f32,
};

fn half_bits(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_bits().to_le_bytes()
}

fn q4_block(d: f32, dmin: f32, scales: [u8; 12], qs_seed: u8) -> [u8; 144] {
    let mut b = [0u8; 144];
    b[0..2].copy_from_slice(&half_bits(d));
    b[2..4].copy_from_slice(&half_bits(dmin));
    b[4..16].copy_from_slice(&scales);
    b[16] = qs_seed;
    b
}

fn q6_block(d: f32, scale: i8, ql0: u8) -> [u8; 210] {
    let mut b = [0u8; 210];
    b[0] = ql0;
    for s in b[192..208].iter_mut() {
        *s = scale as u8;
    }
    b[208..210].copy_from_slice(&half_bits(d));
    b
}

#[test]
fn q4_k_scales_and_mins() {
    let scales = [1, 1, 1, 1, 0, 0, 0, 0, 0x11, 0x11, 0x11, 0x11];
    let y = dequant_q4_k(&q4_block(1.0, 0.0, scales, 0x21));
    assert_eq!(y[0], 1.0);
    assert_eq!(y[1], 0.0);
    assert_eq!(y[32], 2.0);
    assert_eq!(y[33], 0.0);
    assert_eq!(y[64], 0.0);
    assert_eq!(y[128], 0.0);

    let scales_min = [1, 1, 1, 1, 1, 1, 1, 1, 0x11, 0x11, 0x11, 0x11];
    let y = dequant_q4_k(&q4_block(1.0, 1.0, scales_min, 0x21));
    assert_eq!(y[0], 0.0);
    assert_eq!(y[1], -1.0);
    assert_eq!(y[32], 1.0);
    assert_eq!(y[33], -1.0);
}

#[test]
fn q6_k_offsets_and_nibbles() {
    let y = dequant_q6_k(&q6_block(1.0, 1, 0));
    assert!(y.iter().all(|v| *v == -32.0));

    let y = dequant_q6_k(&q6_block(1.0, 1, 0xFF));
    assert_eq!(y[0], -17.0);
    assert_eq!(y[32], -32.0);
    assert_eq!(y[64], -17.0);
    assert_eq!(y[96], -32.0);
    assert_eq!(y[128], -32.0);
}

#[test]
fn q8_0_values() {
    let mut b = [0u8; 34];
    b[0..2].copy_from_slice(&half_bits(1.0));
    for i in 0..32 {
        b[2 + i] = (i as i32 - 10) as i8 as u8;
    }
    let y = dequant_q8_0(&b);
    for (i, v) in y.iter().enumerate().take(32) {
        assert_eq!(*v, (i as i32 - 10) as f32);
    }
}

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn pseudo_q4_blocks(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * 144);
    for _ in 0..n {
        let mut b = [0u8; 144];
        b[0..2].copy_from_slice(&half_bits(0.001 + (lcg(&mut s) % 100) as f32 * 1e-5));
        b[2..4].copy_from_slice(&half_bits((lcg(&mut s) % 100) as f32 * 1e-5));
        for v in b.iter_mut().take(16).skip(4) {
            *v = lcg(&mut s) as u8;
        }
        for v in b.iter_mut().take(144).skip(16) {
            *v = lcg(&mut s) as u8;
        }
        out.extend_from_slice(&b);
    }
    out
}

fn pseudo_q6_blocks(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * 210);
    for _ in 0..n {
        let mut b = [0u8; 210];
        b[208..210].copy_from_slice(&half_bits(0.001 + (lcg(&mut s) % 100) as f32 * 1e-5));
        for v in b.iter_mut().take(208) {
            *v = lcg(&mut s) as u8;
        }
        out.extend_from_slice(&b);
    }
    out
}

#[test]
fn dot_matches_dequant_q4_k() {
    let bytes = pseudo_q4_blocks(4, 7);
    let mut s = 11u64;
    let x: Vec<f32> = (0..1024)
        .map(|_| (lcg(&mut s) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let y = dequant_row_q4_k(&bytes, 1024).unwrap();
    let reference: f32 = y.iter().zip(&x).map(|(a, b)| a * b).sum();
    let dot = dot_row_q4_k(&bytes, &x).unwrap();
    assert!((dot - reference).abs() <= 1e-3 * reference.abs().max(1.0));
}

#[test]
fn dot_matches_dequant_q6_k() {
    let bytes = pseudo_q6_blocks(4, 13);
    let mut s = 17u64;
    let x: Vec<f32> = (0..1024)
        .map(|_| (lcg(&mut s) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let y = dequant_row_q6_k(&bytes, 1024).unwrap();
    let reference: f32 = y.iter().zip(&x).map(|(a, b)| a * b).sum();
    let dot = dot_row_q6_k(&bytes, &x).unwrap();
    assert!((dot - reference).abs() <= 1e-3 * reference.abs().max(1.0));
}

#[test]
fn rejects_bad_lengths() {
    let bytes = pseudo_q4_blocks(1, 3);
    assert!(dequant_row_q4_k(&bytes, 512).is_err());
    assert!(dequant_row_q4_k(&bytes[..100], 256).is_err());
    assert!(dot_row_q6_k(&bytes, &[0.0f32; 128]).is_err());
}

#[test]
fn f16_conversion() {
    let bytes = half_bits(1.5).to_vec();
    assert_eq!(f16s_to_f32(&bytes), vec![1.5f32]);
}

#[test]
fn q4_k_golden_nibble_layout() {
    // d=1, dmin=0, все sc=1, все m=0 => значение каждого элемента равно своему нибблу.
    // Ожидаемый порядок: группа j (0..8) из 32 значений, j = 2*qq + hi;
    // байты группы лежат в qs[qq*32 + l], младший ниббл — j чётный, старший — j нечётный.
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&half_bits(1.0));
    block[2..4].copy_from_slice(&half_bits(0.0));
    block[4..8].copy_from_slice(&[1, 1, 1, 1]);
    block[8..12].copy_from_slice(&[0, 0, 0, 0]);
    block[12..16].copy_from_slice(&[1, 1, 1, 1]);
    for i in 0..128usize {
        let lo = (i % 15) as u8;
        let hi = 15 - lo;
        block[16 + i] = lo | (hi << 4);
    }
    let y = dequant_q4_k(&block);
    for j in 0..8usize {
        let qq = j / 2;
        let high = j % 2 == 1;
        for l in 0..32usize {
            let byte = block[16 + qq * 32 + l];
            let want = if high {
                (byte >> 4) as f32
            } else {
                (byte & 0x0F) as f32
            };
            assert_eq!(y[j * 32 + l], want, "j={j} l={l}");
        }
    }
    let row = dequant_row_q4_k(&block, 256).unwrap();
    assert_eq!(row, y);
}

#[test]
fn q4_k_zero_block_is_zero() {
    let block = q4_block(0.0, 0.0, [0u8; 12], 0xFF);
    let y = dequant_q4_k(&block);
    assert!(y.iter().all(|v| *v == 0.0), "zero block must stay zero");
}
