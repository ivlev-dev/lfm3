//! HVX FastRPC method table.
//!
//! Method IDs and scalar packs are transcribed from the `qaic`-generated
//! `myelin/dsp/hvx/lfmgemv_stub.c` (SDK 6.7.0.0). `build.rs` re-parses
//! `myelin/dsp/hvx/lfmgemv.idl` (and the generated stub, when present) and
//! fails the build if these constants drift. The IDL is the source of truth.
//!
//! Wire layout follows `REMOTE_SCALARS_MAKEX(0, method, in, out, 0, 0)`:
//! `prim_in`, input sequences, `prim_out` (scalar outputs), output sequences.
//! All scalars are little-endian; both the host and Hexagon are little-endian.

/// Method IDs of the `lfmgemv` interface, in IDL order.
pub mod method_id {
    /// `alloc_buf(in long nbytes, rout long h)`
    pub const ALLOC_BUF: u32 = 0;
    /// `set_shape(in long h, in long N, in long K, in long with_scale)`
    pub const SET_SHAPE: u32 = 1;
    /// `load_buf(in long h, in long off, in sequence<octet> data)`
    pub const LOAD_BUF: u32 = 2;
    /// `set_scale(in long h, in sequence<octet> scales)`
    pub const SET_SCALE: u32 = 3;
    /// `gemv(in long h, in long M, in sequence<octet> act, in long reps, ...)`
    pub const GEMV: u32 = 4;
    /// `probe(in long h, in long kind, in long arg, rout sequence<long> res)`
    pub const PROBE: u32 = 5;
    /// `ffn(in long hg, in long hu, in long hd, in sequence<octet> act, ...)`
    pub const FFN: u32 = 6;
    /// `attach_out(in long h, in long fd, in long nbytes)`
    pub const ATTACH_OUT: u32 = 7;
    /// `detach_out(in long h)`
    pub const DETACH_OUT: u32 = 8;
    /// `gemv_resident(in long h, in long M, in sequence<octet> act, ...)`
    pub const GEMV_RESIDENT: u32 = 9;
    /// `ffn_resident(in long hg, in long hu, in long hd, ...)`
    pub const FFN_RESIDENT: u32 = 10;
    /// `attn(in long hk, in long hkt, in long hv, in long hvs, ...)`
    pub const ATTN: u32 = 11;
    /// `team_info(rout sequence<long> res)`
    pub const TEAM_INFO: u32 = 12;
    /// `release_buf(in long h)`
    pub const RELEASE_BUF: u32 = 13;
}

macro_rules! scalar_pack {
    ($name:ident { $($field:ident: $ty:ty),+ $(,)? } size = $size:expr) => {
        /// Scalar input pack (`in long ...`).
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name {
            $(pub $field: $ty,)+
        }

        impl $name {
            /// Pack the fields exactly as `qaic` lays them out.
            pub fn to_bytes(&self) -> [u8; $size] {
                let mut out = [0u8; $size];
                let mut offset = 0;
                $(
                    out[offset..offset + std::mem::size_of::<$ty>()]
                        .copy_from_slice(&self.$field.to_le_bytes());
                    offset += std::mem::size_of::<$ty>();
                )+
                debug_assert_eq!(offset, $size);
                out
            }
        }
    };
}

scalar_pack!(AllocBufScalars { nbytes: i32 } size = 4);
scalar_pack!(AllocBufOut { h: i32 } size = 4);
scalar_pack!(SetShapeScalars {
    h: i32,
    n: i32,
    k: i32,
    with_scale: i32
} size = 16);
scalar_pack!(LoadBufScalars { h: i32, off: i32 } size = 8);
scalar_pack!(SetScaleScalars { h: i32 } size = 4);
scalar_pack!(GemvScalars {
    h: i32,
    m: i32,
    reps: i32
} size = 12);
scalar_pack!(FfnScalars {
    gate: i32,
    up: i32,
    down: i32,
    reps: i32
} size = 16);
scalar_pack!(AttnScalars {
    keys: i32,
    key_scales: i32,
    values: i32,
    value_scales: i32,
    position: i32,
    seq: i32,
    window: i32,
    reps: i32
} size = 32);
scalar_pack!(AttachOutScalars {
    h: i32,
    fd: i32,
    nbytes: i32
} size = 12);
scalar_pack!(DetachOutScalars { h: i32 } size = 4);
scalar_pack!(ReleaseBufScalars { h: i32 } size = 4);
scalar_pack!(TeamInfoScalars { len: i32 } size = 4);

const _: () = {
    assert!(std::mem::size_of::<AllocBufScalars>() == 4);
    assert!(std::mem::size_of::<AllocBufOut>() == 4);
    assert!(std::mem::size_of::<SetShapeScalars>() == 16);
    assert!(std::mem::size_of::<LoadBufScalars>() == 8);
    assert!(std::mem::size_of::<SetScaleScalars>() == 4);
    assert!(std::mem::size_of::<GemvScalars>() == 12);
    assert!(std::mem::size_of::<FfnScalars>() == 16);
    assert!(std::mem::size_of::<AttnScalars>() == 32);
    assert!(std::mem::size_of::<AttachOutScalars>() == 12);
    assert!(std::mem::size_of::<DetachOutScalars>() == 4);
    assert!(std::mem::size_of::<ReleaseBufScalars>() == 4);
    assert!(std::mem::size_of::<TeamInfoScalars>() == 4);
};

/// Typed `gemv` request.
pub struct GemvRequest<'a> {
    pub scalars: GemvScalars,
    /// `M * K` int8 activations.
    pub act: &'a [u8],
    /// Expected `M * N` int8 outputs.
    pub dst_len: usize,
}

/// Typed `gemv` response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemvResponse {
    pub dst: Vec<u8>,
    /// `{min, max, avg, init, qtimer_freq, mac_lo, rows, threads}`.
    pub times: [i32; 8],
}

/// Typed one-RPC FFN request.
pub struct FfnRequest<'a> {
    pub scalars: FfnScalars,
    /// `K` int8 activations.
    pub act: &'a [u8],
    /// Expected final int8 output length (`K`).
    pub dst_len: usize,
    /// Expected SiLU mid-buffer length (`N`).
    pub mid_len: usize,
}

/// Typed one-RPC FFN response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnResponse {
    pub dst: Vec<u8>,
    pub mid: Vec<u8>,
    pub times: [i32; 8],
}

/// Typed paged-attention request.
pub struct AttnRequest<'a> {
    pub scalars: AttnScalars,
    /// `16 * 128` int8 query.
    pub q: &'a [u8],
    /// Block table, one page id per 32 tokens.
    pub table: &'a [i32],
    /// Expected `16 * 128 * 4` int32 output bytes.
    pub out_len: usize,
}

/// Typed paged-attention response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttnResponse {
    /// `16 * 128` int32 Q12 values, little-endian.
    pub outp: Vec<u8>,
    pub times: [i32; 8],
}

/// Decode an `rout sequence<long>` buffer.
pub fn decode_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
