#!/usr/bin/env python3
"""Compare two --logits dumps from q4k_host: cos, max|d|, top-5 agreement."""
import sys
import numpy as np

ref = np.fromfile(sys.argv[1], dtype=np.float32)
for p in sys.argv[2:]:
    y = np.fromfile(p, dtype=np.float32)
    n = min(len(ref), len(y))
    a, b = ref[:n].astype(np.float64), y[:n].astype(np.float64)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    d = np.abs(a - b)
    t5r = set(np.argsort(-a)[:5].tolist())
    t5y = set(np.argsort(-b)[:5].tolist())
    print("%-28s cos=%.9f  max|dlogit|=%.4f  top1 %s/%s  top5 overlap=%d/5"
          % (p.split('/')[-1], cos, d.max(),
             int(np.argmax(a)), int(np.argmax(b)), len(t5r & t5y)))
