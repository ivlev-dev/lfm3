import struct, json, sys
import numpy as np

PATH = 'models/LFM3-1.7B-Q4_K_M.gguf'
f = open(PATH, 'rb')

def rd(n):
    b = f.read(n)
    assert len(b) == n
    return b

def u32(): return struct.unpack('<I', rd(4))[0]
def u64(): return struct.unpack('<Q', rd(8))[0]
def rs(): return rd(u64()).decode('utf-8', 'replace')

FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}

def val(t):
    if t == 8: return rs()
    if t == 9:
        et = u32(); n = u64()
        if et == 8:
            if n > 1000:
                for _ in range(n): rd(u64())
                return None
            return [rs() for _ in range(n)]
        if et == 9:
            raise SystemExit('nested array')
        sz = struct.calcsize(FMT[et])
        return np.frombuffer(rd(sz * n), dtype=FMT[et]).tolist()
    return struct.unpack(FMT[t], rd(struct.calcsize(FMT[t])))[0]

rd(4); u32(); nt = u64(); nkv = u64()
meta = {}
for _ in range(nkv):
    k = rs(); t = u32(); meta[k] = val(t)
align = meta.get('general.alignment', 32)
infos = []
for _ in range(nt):
    n = rs(); nd = u32(); dims = [u64() for _ in range(nd)]; tt = u32(); off = u64()
    infos.append((n, dims, tt, off))
data_off = (f.tell() + align - 1) // align * align
INFO = {n: (dims, tt, off) for n, dims, tt, off in infos}

def raw(name):
    dims, tt, off = INFO[name]
    f.seek(data_off + off)
    count = 1
    for d in dims: count *= d
    sizes = {0:4, 1:2, 30:2, 12:None, 14:None, 8:None}
    if tt == 0: return rd(count * 4), tt, dims
    if tt == 1: return rd(count * 2), tt, dims
    if tt == 12: return rd(count // 256 * 144), tt, dims
    if tt == 14: return rd(count // 256 * 210), tt, dims
    if tt == 8: return rd(count // 32 * 34), tt, dims
    raise SystemExit('unsupported type %d for %s' % (tt, name))

def gsm4(sc, j):
    if j < 4:
        return sc[:, j] & 63, sc[:, j + 4] & 63
    d = ((sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)).astype(np.uint8)
    m = ((sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)).astype(np.uint8)
    return d, m

def dequant_q4k(buf, nblocks):
    b = np.frombuffer(buf, dtype=np.uint8).reshape(nblocks, 144)
    d = b[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).reshape(-1).astype(np.float32)
    sc = b[:, 4:16]
    qs = b[:, 16:144]
    y = np.zeros((nblocks, 256), dtype=np.float32)
    for g in range(4):
        q = qs[:, g*32:(g+1)*32].astype(np.int32)
        for half in range(2):
            j = 2 * g + half
            s, m = gsm4(sc, j)
            vals = (q & 0xF) if half == 0 else (q >> 4)
            base = g * 64 + half * 32
            y[:, base:base+32] = d[:, None] * s[:, None].astype(np.float32) * vals - dmin[:, None] * m[:, None].astype(np.float32)
    return y.reshape(-1)

def dequant_q6k(buf, nblocks):
    b = np.frombuffer(buf, dtype=np.uint8).reshape(nblocks, 210)
    d = b[:, 208:210].copy().view(np.float16).reshape(-1).astype(np.float32)
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.int32)
    y = np.zeros((nblocks, 256), dtype=np.float32)
    for n in range(2):
        for l in range(32):
            base = n * 64
            q1 = ((ql[:, base + l] & 0xF) | (((qh[:, n*32 + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[:, base + l + 32] & 0xF) | (((qh[:, n*32 + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[:, base + l] >> 4) | (((qh[:, n*32 + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[:, base + l + 32] >> 4) | (((qh[:, n*32 + l] >> 6) & 3) << 4)) - 32
            isb = l // 16
            y[:, n*128 + l + 0] = d * sc[:, n*8 + isb + 0] * q1
            y[:, n*128 + l + 32] = d * sc[:, n*8 + isb + 2] * q2
            y[:, n*128 + l + 64] = d * sc[:, n*8 + isb + 4] * q3
            y[:, n*128 + l + 96] = d * sc[:, n*8 + isb + 6] * q4
    return y.reshape(-1)

CACHE = {}

def tensor(name):
    if name in CACHE: return CACHE[name]
    buf, tt, dims = raw(name)
    count = 1
    for d in dims: count *= d
    if tt == 0:
        arr = np.frombuffer(buf, dtype='<f4').astype(np.float32)
    elif tt == 1:
        arr = np.frombuffer(buf, dtype='<f2').astype(np.float32)
    elif tt == 12:
        arr = dequant_q4k(buf, count // 256)
    elif tt == 14:
        arr = dequant_q6k(buf, count // 256)
    elif tt == 8:
        b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 34)
        d = b[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
        q = b[:, 2:34].copy().view(np.int8).astype(np.float32)
        arr = (d[:, None] * q).reshape(-1)
    else:
        raise SystemExit('type')
    if len(dims) == 2:
        arr = arr.reshape(dims[1], dims[0])
    CACHE[name] = arr
    return arr

H = meta['lfm3.embedding_length']
FF = meta['lfm3.feed_forward_length']
NL = meta['lfm3.block_count']
NH = meta['lfm3.attention.head_count']
KVH = meta['lfm3.attention.head_count_kv']
SWA = meta['lfm3.attention.sliding_window_pattern']
HD = meta['lfm3.attention.key_length']
EPS = float(meta['lfm3.attention.layer_norm_rms_epsilon'])
CONV = meta['lfm3.shortconv.l_cache']
WIN = meta['lfm3.attention.sliding_window']
THETA = float(meta['lfm3.rope.freq_base'])
THETA_SWA = float(meta['lfm3.rope.freq_base_swa'])
VOCAB = tensor('token_embd.weight').shape[0]
KIND = ['conv' if KVH[i] == 0 else ('swa' if SWA[i] else 'full') for i in range(NL)]

def inv_freq(theta, rotated):
    half = HD // 2
    out = np.zeros(half, dtype=np.float32)
    rot = rotated // 2
    for i in range(rot):
        out[i] = 1.0 / (theta ** ((2.0 * i) / HD))
    return out

INV_FULL = inv_freq(THETA, HD // 2)
INV_SWA = inv_freq(THETA_SWA, HD)

def rms(x, w):
    var = np.mean(np.square(x, dtype=np.float64))
    return (x * np.float32(1.0 / np.sqrt(var + EPS)) * w).astype(np.float32)

def silu(x): return x / (1.0 + np.exp(-x))

VM = np.vectorize(lambda i: 1 if i == 597 else 0)

class State:
    def __init__(self):
        self.conv = [np.zeros((CONV - 1) * H, dtype=np.float32) for _ in range(NL)]
        self.k = [None] * NL
        self.v = [None] * NL
    def append(self, i, k, v):
        if self.k[i] is None:
            self.k[i] = k[None, :]; self.v[i] = v[None, :]
        else:
            self.k[i] = np.concatenate([self.k[i], k[None, :]], axis=0)
            self.v[i] = np.concatenate([self.v[i], v[None, :]], axis=0)

def rope(x, pos, inv):
    half = HD // 2
    out = x.copy()
    for i in range(half):
        if inv[i] == 0.0: continue
        ang = pos * inv[i]
        c, s = np.cos(ang), np.sin(ang)
        a, b = x[i], x[i + half]
        out[i] = a * c - b * s
        out[i + half] = b * c + a * s
    return out.astype(np.float32)

def forward(st, token, pos):
    x = tensor('token_embd.weight')[token].copy()
    for i in range(NL):
        p = 'blk.%d.' % i
        residual = x
        h = rms(x, tensor(p + 'attn_norm.weight'))
        if KIND[i] == 'conv':
            bcx = tensor(p + 'shortconv.in_proj.weight') @ h
            b, c, xv = bcx[:H], bcx[H:2*H], bcx[2*H:]
            bx = (b * xv).astype(np.float32)
            st.conv[i] = np.concatenate([st.conv[i], bx])[-(CONV - 1) * H:]
            win = np.concatenate([st.conv[i].reshape(CONV - 1, H), bx[None, :]], axis=0)
            w = tensor(p + 'shortconv.conv.weight')
            out = np.zeros(H, dtype=np.float32)
            for t in range(CONV):
                out += w[:, t] * win[t]
            op = tensor(p + 'shortconv.out_proj.weight') @ (c * out)
        else:
            q = tensor(p + 'attn_q.weight') @ h
            k = tensor(p + 'attn_k.weight') @ h
            v = tensor(p + 'attn_v.weight') @ h
            qn = tensor(p + 'attn_q_norm.weight')
            kn = tensor(p + 'attn_k_norm.weight')
            inv = INV_SWA if KIND[i] == 'swa' else INV_FULL
            for hh in range(NH):
                s = hh * HD
                q[s:s+HD] = rope(rms(q[s:s+HD], qn), pos, inv)
            kvh = KVH[i]
            for hh in range(kvh):
                s = hh * HD
                k[s:s+HD] = rope(rms(k[s:s+HD], kn), pos, inv)
            st.append(i, k, v)
            K = st.k[i]; V = st.v[i]
            total = K.shape[0]
            window = min(WIN, total) if KIND[i] == 'swa' else total
            start = total - window
            scale = np.float32(1.0 / np.sqrt(HD))
            attn = np.zeros((NH, HD), dtype=np.float32)
            for hh in range(NH):
                kv = hh // (NH // kvh)
                qh = q[hh*HD:(hh+1)*HD]
                Ks = K[start:total, kv*HD:(kv+1)*HD]
                scores = (Ks @ qh) * scale
                e = np.exp(scores - scores.max())
                probs = e / e.sum()
                attn[hh] = probs @ V[start:total, kv*HD:(kv+1)*HD]
            g = 1.0 / (1.0 + np.exp(-(tensor(p + 'attn_gate.weight') @ h)))
            attn = attn * g[:, None]
            op = tensor(p + 'attn_output.weight') @ attn.reshape(-1)
        x = rms(op, tensor(p + 'post_attention_norm.weight')) + residual
        ffn_in = rms(x, tensor(p + 'ffn_norm.weight'))
        up = tensor(p + 'ffn_up.weight') @ ffn_in
        gate = tensor(p + 'ffn_gate.weight') @ ffn_in
        act = (silu(gate) * up).astype(np.float32)
        ffn = tensor(p + 'ffn_down.weight') @ act
        x = x + rms(ffn, tensor(p + 'post_ffw_norm.weight'))
    x = rms(x, tensor('output_norm.weight'))
    logits = tensor('token_embd.weight') @ x
    return logits

def generate(ids, n):
    st = State()
    logits = None
    for pos, t in enumerate(ids):
        logits = forward(st, t, pos)
    out = []
    for pos in range(len(ids), len(ids) + n):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        logits = forward(st, nxt, pos)
    return out, logits

if __name__ == '__main__':
    ids = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [597]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out, logits = generate(ids, n)
    print('top5', np.argsort(-logits)[:5].tolist())
    print('gen', out)
    print('kinds', KIND)
