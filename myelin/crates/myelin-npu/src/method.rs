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
    /// `attach_in(in long h, in long fd, in long nbytes)`
    pub const ATTACH_IN: u32 = 14;
    /// `detach_in(in long h)`
    pub const DETACH_IN: u32 = 15;
    /// `alloc_in(in long fd, in long nbytes, rout long h)`
    pub const ALLOC_IN: u32 = 16;
    /// `map_arena(in long fd, in long nbytes, rout long ah)`
    pub const MAP_ARENA: u32 = 17;
    /// `unmap_arena(in long ah)`
    pub const UNMAP_ARENA: u32 = 18;
    /// `alloc_view(in long ah, in long off, in long nbytes, rout long h)`
    pub const ALLOC_VIEW: u32 = 19;
    /// `reserve_scratch(in long nbytes)`
    pub const RESERVE_SCRATCH: u32 = 20;
    /// `exec(in long ah, in long cmd_off, in long n_cmds, ...)`
    pub const EXEC: u32 = 21;
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

// The scalar packs below are the *wire* layout of `_primIn[]` in
// `lfmgemv_stub.c`. qaic writes every sequence length into that array as well
// (`_COPY(_primIn, 8, _in2Len, 0, 4)` for `load_buf`), so a pack that carries
// only the declared `in long` arguments is not accepted by the skeleton
// (`AEE_EBADPARM`). `PRIM_IN_SIZES` is checked against the generated stub by
// `build.rs`.
scalar_pack!(AllocBufScalars { nbytes: i32 } size = 4);
scalar_pack!(AllocBufOut { h: i32 } size = 4);
scalar_pack!(SetShapeScalars {
    h: i32,
    n: i32,
    k: i32,
    with_scale: i32
} size = 16);
scalar_pack!(LoadBufWire {
    h: i32,
    off: i32,
    data_len: i32
} size = 12);
scalar_pack!(SetScaleWire { h: i32, data_len: i32 } size = 8);
scalar_pack!(GemvWire {
    h: i32,
    m: i32,
    act_len: i32,
    reps: i32,
    dst_len: i32,
    times_len: i32
} size = 24);
scalar_pack!(FfnWire {
    gate: i32,
    up: i32,
    down: i32,
    act_len: i32,
    reps: i32,
    dst_len: i32,
    mid_len: i32,
    times_len: i32
} size = 32);
scalar_pack!(ProbeWire {
    h: i32,
    kind: i32,
    arg: i32,
    res_len: i32
} size = 16);
scalar_pack!(GemvResidentWire {
    h: i32,
    m: i32,
    act_len: i32,
    reps: i32,
    times_len: i32
} size = 20);
scalar_pack!(FfnResidentWire {
    gate: i32,
    up: i32,
    down: i32,
    act_len: i32,
    reps: i32,
    mid_len: i32,
    times_len: i32
} size = 28);
scalar_pack!(AttnWire {
    keys: i32,
    key_scales: i32,
    values: i32,
    value_scales: i32,
    q_len: i32,
    tbl_len: i32,
    position: i32,
    seq: i32,
    window: i32,
    reps: i32,
    out_len: i32,
    times_len: i32
} size = 48);
scalar_pack!(AttachOutScalars {
    h: i32,
    fd: i32,
    nbytes: i32
} size = 12);
scalar_pack!(DetachOutScalars { h: i32 } size = 4);
scalar_pack!(ReleaseBufScalars { h: i32 } size = 4);
scalar_pack!(TeamInfoScalars { len: i32 } size = 4);
scalar_pack!(AttachInScalars {
    h: i32,
    fd: i32,
    nbytes: i32
} size = 12);
scalar_pack!(DetachInScalars { h: i32 } size = 4);
scalar_pack!(AllocInScalars { fd: i32, nbytes: i32 } size = 8);
scalar_pack!(AllocInOut { h: i32 } size = 4);
scalar_pack!(MapArenaWire { fd: i32, nbytes: i32 } size = 8);
scalar_pack!(ArenaOut { ah: i32 } size = 4);
scalar_pack!(UnmapArenaScalars { ah: i32 } size = 4);
scalar_pack!(AllocViewWire { ah: i32, off: i32, nbytes: i32 } size = 12);
scalar_pack!(ReserveScratchScalars { nbytes: i32 } size = 4);
scalar_pack!(ExecWire {
    ah: i32,
    cmd_off: i32,
    n_cmds: i32,
    act_len: i32,
    rt_len: i32,
    outp_len: i32,
    times_len: i32
} size = 28);

/// `sizeof(_primIn)` per method, in method-ID order (`lfmgemv_stub.c`).
pub const PRIM_IN_SIZES: [usize; 22] = [
    4, 16, 12, 8, 24, 16, 32, 12, 4, 20, 28, 48, 4, 4, 12, 4, 8, 8, 4, 12, 4, 28,
];

/// Number of i32 values in the stand's `times[]` sequence.
pub const TIMES_VALUES: usize = 8;

const _: () = {
    assert!(std::mem::size_of::<AllocBufScalars>() == PRIM_IN_SIZES[0]);
    assert!(std::mem::size_of::<SetShapeScalars>() == PRIM_IN_SIZES[1]);
    assert!(std::mem::size_of::<LoadBufWire>() == PRIM_IN_SIZES[2]);
    assert!(std::mem::size_of::<SetScaleWire>() == PRIM_IN_SIZES[3]);
    assert!(std::mem::size_of::<GemvWire>() == PRIM_IN_SIZES[4]);
    assert!(std::mem::size_of::<ProbeWire>() == PRIM_IN_SIZES[5]);
    assert!(std::mem::size_of::<FfnWire>() == PRIM_IN_SIZES[6]);
    assert!(std::mem::size_of::<AttachOutScalars>() == PRIM_IN_SIZES[7]);
    assert!(std::mem::size_of::<DetachOutScalars>() == PRIM_IN_SIZES[8]);
    assert!(std::mem::size_of::<GemvResidentWire>() == PRIM_IN_SIZES[9]);
    assert!(std::mem::size_of::<FfnResidentWire>() == PRIM_IN_SIZES[10]);
    assert!(std::mem::size_of::<AttnWire>() == PRIM_IN_SIZES[11]);
    assert!(std::mem::size_of::<TeamInfoScalars>() == PRIM_IN_SIZES[12]);
    assert!(std::mem::size_of::<ReleaseBufScalars>() == PRIM_IN_SIZES[13]);
    assert!(std::mem::size_of::<AttachInScalars>() == PRIM_IN_SIZES[14]);
    assert!(std::mem::size_of::<DetachInScalars>() == PRIM_IN_SIZES[15]);
    assert!(std::mem::size_of::<AllocInScalars>() == PRIM_IN_SIZES[16]);
    assert!(std::mem::size_of::<AllocBufOut>() == 4);
    assert!(std::mem::size_of::<AllocInOut>() == 4);
    assert!(std::mem::size_of::<MapArenaWire>() == PRIM_IN_SIZES[17]);
    assert!(std::mem::size_of::<UnmapArenaScalars>() == PRIM_IN_SIZES[18]);
    assert!(std::mem::size_of::<AllocViewWire>() == PRIM_IN_SIZES[19]);
    assert!(std::mem::size_of::<ReserveScratchScalars>() == PRIM_IN_SIZES[20]);
    assert!(std::mem::size_of::<ExecWire>() == PRIM_IN_SIZES[21]);
    assert!(std::mem::size_of::<ArenaOut>() == 4);
};

/// `sequence<octet>` length in elements (one element is one byte).
pub fn octet_len(bytes: usize) -> Option<i32> {
    i32::try_from(bytes).ok()
}

/// `sequence<long>` length in elements (`4` bytes each).
pub fn long_len(bytes: usize) -> Option<i32> {
    if bytes % 4 != 0 {
        return None;
    }
    i32::try_from(bytes / 4).ok()
}

/// Declared `in long` arguments of `gemv`; the sequence lengths are derived
/// from the buffers in [`GemvRequest`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GemvScalars {
    pub h: i32,
    pub m: i32,
    pub reps: i32,
}

/// Declared `in long` arguments of `ffn`; lengths come from the request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FfnScalars {
    pub gate: i32,
    pub up: i32,
    pub down: i32,
    pub reps: i32,
}

/// Declared `in long` arguments of `attn`; lengths come from the request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttnScalars {
    pub keys: i32,
    pub key_scales: i32,
    pub values: i32,
    pub value_scales: i32,
    pub position: i32,
    pub seq: i32,
    pub window: i32,
    pub reps: i32,
}

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

/// One `exec` invocation: the command list already lives in `arena` at
/// `cmd_off` (see [`exec::List`]); only activations and per-token values are
/// sent through the RPC.
pub struct ExecRequest<'a> {
    pub arena: i32,
    pub cmd_off: i32,
    /// Number of commands; must equal the list header's `n_cmds`.
    pub commands: usize,
    /// Initial activations, copied into scratch at offset 0.
    pub act: &'a [u8],
    /// Per-token values referenced by `rt_base` (pos, seq, window, reps).
    pub rt: &'a [i32],
    /// Size of the result area for commands with `dst = OUT`.
    pub out_len: usize,
}

/// `exec` response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResponse {
    pub out: Vec<u8>,
    /// `{dsp_us, failed_command or -1, commands_run, team_fallbacks}`.
    pub times: [i32; 8],
}

impl ExecResponse {
    /// Index of the command that failed, or `None`.
    pub const fn failed_command(&self) -> Option<usize> {
        if self.times[1] < 0 {
            None
        } else {
            Some(self.times[1] as usize)
        }
    }
}

/// Decode an `rout sequence<long>` buffer.
pub fn decode_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// The versioned command list consumed by `exec`, byte-compatible with
/// `lfm_exec_hdr` / `lfm_cmd` in `myelin/dsp/hvx/lfmgemv_impl.c`.
///
/// The list is written into an ION buffer *before* `map_arena` maps it: the
/// DSP caches mapped pages, so the list must not change while mapped.  Only
/// the per-token values in `rt` (attention position, sequence, window) and the
/// activation sequence travel through the RPC.
pub mod exec {
    /// `LFM_EXEC_MAGIC`.
    pub const MAGIC: u32 = 0x454D_464C;
    /// `LFM_EXEC_VERSION`.
    pub const VERSION: u32 = 1;
    /// `sizeof(lfm_exec_hdr)`.
    pub const HEADER_BYTES: usize = 64;
    /// `sizeof(lfm_cmd)`.
    pub const COMMAND_BYTES: usize = 96;
    /// `LFM_CMD_MAX`.
    pub const MAX_COMMANDS: usize = 256;

    /// `LFM_SRC_*`: where a command reads its activation from.
    pub mod src {
        /// DSP-heap scratch, at `src_off`.
        pub const SCRATCH: u32 = 0;
        /// The per-call `out` sequence.
        pub const OUT: u32 = 1;
    }

    /// `LFM_DST_*`: where a command writes its result.
    pub mod dst {
        /// DSP-heap scratch, at `dst_off`.
        pub const SCRATCH: u32 = 0;
        /// The per-call `out` sequence.
        pub const OUT: u32 = 1;
        /// `attach_out` buffer of `dst_h`, at `dst_off`.
        pub const ATTACHED: u32 = 2;
    }

    /// Command type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Op {
        /// int32 -> int8 with one Q16 multiplier (`a0`).
        Quant = 0,
        /// GEMV into an attached/scratch/output buffer.
        Gemv = 1,
        /// Fused gate/up -> SiLU -> down.
        Ffn = 2,
        /// Paged attention.
        Attn = 3,
        /// Depthwise shortconv step.
        Conv = 4,
    }

    fn put_i32(out: &mut [u8], offset: usize, value: i32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(out: &mut [u8], offset: usize, value: u32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// One 96-byte command. Defaults are "no handle": `h = dst_h = tbl_h = -1`,
    /// `rt_base = -1`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Command {
        pub op: Op,
        pub src: u32,
        pub dst: u32,
        pub h: [i32; 4],
        pub dst_h: i32,
        pub src_off: i32,
        pub src_len: i32,
        pub dst_off: i32,
        pub dst_len: i32,
        pub tbl_h: i32,
        pub tbl_len: i32,
        pub tbl_off: i32,
        pub args: [i32; 4],
        pub rt_base: i32,
    }

    impl Command {
        const fn empty(op: Op) -> Self {
            Self {
                op,
                src: src::SCRATCH,
                dst: dst::SCRATCH,
                h: [-1; 4],
                dst_h: -1,
                src_off: 0,
                src_len: 0,
                dst_off: 0,
                dst_len: 0,
                tbl_h: -1,
                tbl_len: 0,
                tbl_off: 0,
                args: [0; 4],
                rt_base: -1,
            }
        }

        /// `QUANT`: `n` int32 values at `src_off` -> `n` int8 at `dst_off`.
        pub const fn quant(src_off: i32, dst_off: i32, n: i32, multiplier_q16: i32) -> Self {
            let mut command = Self::empty(Op::Quant);
            command.src_off = src_off;
            command.src_len = n * 4;
            command.dst_off = dst_off;
            command.dst_len = n;
            command.args[0] = multiplier_q16;
            command
        }

        /// `GEMV`: `h` x (`m * k` int8 at `src_off`) -> `m * n` int8 at `dst_off`.
        pub const fn gemv(
            h: i32,
            m: i32,
            reps: i32,
            src_off: i32,
            src_len: i32,
            dst_off: i32,
            dst_len: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Gemv);
            command.h[0] = h;
            command.args[0] = m;
            command.args[1] = reps;
            command.src_off = src_off;
            command.src_len = src_len;
            command.dst_off = dst_off;
            command.dst_len = dst_len;
            command
        }

        /// `FFN`: gate/up/down, `k` int8 in -> `n` int8 out.
        #[allow(clippy::too_many_arguments)]
        pub const fn ffn(
            gate: i32,
            up: i32,
            down: i32,
            reps: i32,
            src_off: i32,
            src_len: i32,
            dst_off: i32,
            dst_len: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Ffn);
            command.h = [gate, up, down, -1];
            command.args[0] = reps;
            command.src_off = src_off;
            command.src_len = src_len;
            command.dst_off = dst_off;
            command.dst_len = dst_len;
            command
        }

        /// `ATTN`: q at `src_off`, int32 Q12 at `dst_off`, block table in
        /// handle `tbl_h`. `rt_base >= 0` takes pos/seq/window/reps from
        /// `rt[rt_base..rt_base + 4]`; otherwise `args` carries them.
        #[allow(clippy::too_many_arguments)]
        pub const fn attn(
            keys: i32,
            key_scales: i32,
            values: i32,
            value_scales: i32,
            tbl_h: i32,
            tbl_len: i32,
            rt_base: i32,
            src_off: i32,
            dst_off: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Attn);
            command.h = [keys, key_scales, values, value_scales];
            command.tbl_h = tbl_h;
            command.tbl_len = tbl_len;
            command.rt_base = rt_base;
            command.src_off = src_off;
            command.src_len = 16 * 128;
            command.dst_off = dst_off;
            command.dst_len = 16 * 128 * 4;
            command
        }

        /// `CONV`: quantised shortconv blob in `blob`, ring state in scratch
        /// at `state_off`; `3 * h` int8 in -> `h` int8 out.
        #[allow(clippy::too_many_arguments)]
        pub const fn conv(
            blob: i32,
            h: i32,
            ksh: i32,
            slot: i32,
            counter: i32,
            state_off: i32,
            src_off: i32,
            dst_off: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Conv);
            command.h[0] = blob;
            command.args = [h, ksh, slot, counter];
            command.tbl_off = state_off;
            command.src_off = src_off;
            command.src_len = 3 * h;
            command.dst_off = dst_off;
            command.dst_len = h;
            command
        }

        /// Override the destination selector and offset.
        pub const fn with_dst(mut self, kind: u32, offset: i32) -> Self {
            self.dst = kind;
            self.dst_off = offset;
            self
        }

        /// Set `dst_h` for [`dst::ATTACHED`].
        pub const fn with_dst_handle(mut self, handle: i32) -> Self {
            self.dst_h = handle;
            self
        }

        /// Wire bytes, exactly `lfm_cmd`.
        pub fn to_bytes(&self) -> [u8; COMMAND_BYTES] {
            let mut out = [0u8; COMMAND_BYTES];
            put_u32(&mut out, 0, self.op as u32);
            put_u32(&mut out, 4, self.src);
            put_u32(&mut out, 8, self.dst);
            for (index, handle) in self.h.iter().enumerate() {
                put_i32(&mut out, 16 + 4 * index, *handle);
            }
            put_i32(&mut out, 32, self.dst_h);
            put_i32(&mut out, 36, self.src_off);
            put_i32(&mut out, 40, self.src_len);
            put_i32(&mut out, 44, self.dst_off);
            put_i32(&mut out, 48, self.dst_len);
            put_i32(&mut out, 52, self.tbl_h);
            put_i32(&mut out, 56, self.tbl_len);
            put_i32(&mut out, 60, self.tbl_off);
            for (index, arg) in self.args.iter().enumerate() {
                put_i32(&mut out, 64 + 4 * index, *arg);
            }
            put_i32(&mut out, 80, self.rt_base);
            out
        }
    }

    /// Header + commands of one `exec` list.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct List {
        pub scratch_bytes: u32,
        pub commands: Vec<Command>,
    }

    impl List {
        pub fn new(scratch_bytes: u32) -> Self {
            Self {
                scratch_bytes,
                commands: Vec::new(),
            }
        }

        /// Append a command; panics past [`MAX_COMMANDS`] because the DSP list
        /// is fixed-size and silent truncation would change the graph.
        pub fn push(&mut self, command: Command) {
            assert!(
                self.commands.len() < MAX_COMMANDS,
                "exec list holds at most {MAX_COMMANDS} commands"
            );
            self.commands.push(command);
        }

        /// Number of bytes [`List::to_bytes`] will produce.
        pub fn len(&self) -> usize {
            HEADER_BYTES + self.commands.len() * COMMAND_BYTES
        }

        pub fn is_empty(&self) -> bool {
            self.commands.is_empty()
        }

        /// Header + commands, ready to be written into the arena.
        pub fn to_bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.len()];
            put_u32(&mut out, 0, MAGIC);
            put_u32(&mut out, 4, VERSION);
            put_u32(&mut out, 8, self.commands.len() as u32);
            put_u32(&mut out, 12, COMMAND_BYTES as u32);
            put_u32(&mut out, 16, self.len() as u32);
            put_u32(&mut out, 20, self.scratch_bytes);
            put_u32(&mut out, 28, HEADER_BYTES as u32);
            for (index, command) in self.commands.iter().enumerate() {
                let at = HEADER_BYTES + index * COMMAND_BYTES;
                out[at..at + COMMAND_BYTES].copy_from_slice(&command.to_bytes());
            }
            out
        }
    }
}

#[cfg(test)]
mod exec_tests {
    use super::exec::*;

    #[test]
    fn core_encoder_matches_the_dsp_layout() {
        use myelin_core::backend::dsp::exec as core_exec;

        let mut ours = List::new(1 << 20);
        ours.push(Command::quant(0, 1024, 2048, 4096));
        ours.push(
            Command::gemv(5, 1, 2, 0, 2048, 1024, 7680)
                .with_dst(dst::OUT, 64)
                .with_dst_handle(11),
        );
        ours.push(Command::attn(1, 2, 3, 4, -1, 16, 0, 0, 8192));
        ours.push(Command::conv(9, 2048, 18, 0, 100, 34816, 0, 16384));

        let mut theirs = core_exec::List::new(1 << 20);
        theirs.push(core_exec::Command::quant(0, 1024, 2048, 4096));
        theirs.push(
            core_exec::Command::gemv(5, 1, 2, 0, 2048, 1024, 7680)
                .with_dst(core_exec::dst::OUT, 64)
                .with_dst_handle(11),
        );
        theirs.push(core_exec::Command::attn(1, 2, 3, 4, -1, 16, 0, 0, 8192));
        theirs.push(core_exec::Command::conv(
            9, 2048, 18, 0, 100, 34816, 0, 16384,
        ));
        assert_eq!(ours.to_bytes(), theirs.to_bytes());
    }

    #[test]
    fn command_layout_matches_the_dsp_struct() {
        let command = Command::ffn(11, 12, 13, 2, 0, 2048, 4096, 7680)
            .with_dst(dst::ATTACHED, 64)
            .with_dst_handle(11);
        let bytes = command.to_bytes();
        assert_eq!(&bytes[0..4], &2u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &src::SCRATCH.to_le_bytes());
        assert_eq!(&bytes[8..12], &dst::ATTACHED.to_le_bytes());
        assert_eq!(&bytes[16..20], &11i32.to_le_bytes());
        assert_eq!(&bytes[20..24], &12i32.to_le_bytes());
        assert_eq!(&bytes[24..28], &13i32.to_le_bytes());
        assert_eq!(&bytes[28..32], &(-1i32).to_le_bytes());
        assert_eq!(&bytes[32..36], &11i32.to_le_bytes());
        assert_eq!(&bytes[36..40], &0i32.to_le_bytes());
        assert_eq!(&bytes[40..44], &2048i32.to_le_bytes());
        assert_eq!(&bytes[44..48], &64i32.to_le_bytes());
        assert_eq!(&bytes[48..52], &7680i32.to_le_bytes());
        assert_eq!(&bytes[52..56], &(-1i32).to_le_bytes());
        assert_eq!(&bytes[64..68], &2i32.to_le_bytes());
        assert_eq!(&bytes[80..84], &(-1i32).to_le_bytes());
    }

    #[test]
    fn list_header_is_versioned_and_length_checked() {
        let mut list = List::new(1 << 20);
        list.push(Command::quant(0, 1024, 2048, 4096));
        list.push(Command::gemv(5, 1, 1, 0, 2048, 1024, 7680));
        let bytes = list.to_bytes();
        assert_eq!(bytes.len(), HEADER_BYTES + 2 * COMMAND_BYTES);
        assert_eq!(&bytes[0..4], &MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &VERSION.to_le_bytes());
        assert_eq!(&bytes[8..12], &2u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &(COMMAND_BYTES as u32).to_le_bytes());
        assert_eq!(&bytes[16..20], &(bytes.len() as u32).to_le_bytes());
        assert_eq!(&bytes[20..24], &(1u32 << 20).to_le_bytes());
        assert_eq!(&bytes[28..32], &(HEADER_BYTES as u32).to_le_bytes());
    }

    #[test]
    fn conv_packs_geometry_into_args() {
        let command = Command::conv(9, 2048, 18, 0, 100, 34816, 0, 16384);
        let bytes = command.to_bytes();
        assert_eq!(&bytes[0..4], &4u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &9i32.to_le_bytes());
        assert_eq!(&bytes[36..40], &0i32.to_le_bytes());
        assert_eq!(&bytes[40..44], &6144i32.to_le_bytes());
        assert_eq!(&bytes[44..48], &16384i32.to_le_bytes());
        assert_eq!(&bytes[48..52], &2048i32.to_le_bytes());
        assert_eq!(&bytes[60..64], &34816i32.to_le_bytes());
        assert_eq!(&bytes[64..68], &2048i32.to_le_bytes());
        assert_eq!(&bytes[68..72], &18i32.to_le_bytes());
        assert_eq!(&bytes[72..76], &0i32.to_le_bytes());
        assert_eq!(&bytes[76..80], &100i32.to_le_bytes());
    }
}
