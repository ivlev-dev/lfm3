import argparse
import http.client
import json
import socket
import sys
import threading
import time
import urllib.parse


def parse_url(url):
    p = urllib.parse.urlparse(url)
    scheme = p.scheme or "http"
    host = p.hostname or "localhost"
    port = p.port or (443 if scheme == "https" else 8000)
    path = p.path.rstrip("/") + "/v1/chat/completions"
    if p.query:
        path += "?" + p.query
    return scheme, host, port, path


def worker(idx, args, scheme, host, port, path, results):
    result = {
        "id": idx,
        "ttft": None,
        "tokens": 0,
        "elapsed": 0.0,
        "error": None,
        "timeout": False,
        "content": "",
        "done": False,
        "valid": True,
    }
    start = time.time()
    ttft_abs = None
    try:
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(host, port, timeout=args.timeout)
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": f"Reply with the single word: {idx}"}],
                "max_tokens": args.tokens,
                "stream": not args.nonstream,
                "temperature": 0,
            },
            ensure_ascii=False,
        )
        accept = "application/json" if args.nonstream else "text/event-stream"
        conn.request(
            "POST",
            path,
            body=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": accept,
            },
        )
        resp = conn.getresponse()
        if resp.status != 200:
            result["error"] = f"HTTP {resp.status}"
            conn.close()
            result["elapsed"] = time.time() - start
            results[idx] = result
            return

        if args.nonstream:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict) or "choices" not in obj or not obj["choices"]:
                result["valid"] = False
                result["error"] = "invalid JSON/choices"
            else:
                content = obj["choices"][0].get("message", {}).get("content", "")
                result["content"] = content if isinstance(content, str) else ""
                usage = obj.get("usage") or {}
                ct = usage.get("completion_tokens")
                if isinstance(ct, int) and ct > 0:
                    result["tokens"] = ct
                else:
                    result["tokens"] = len(result["content"].split()) if result["content"].strip() else 1
                if not result["content"].strip():
                    result["error"] = "empty response"
        else:
            collected = []
            done_seen = False
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8")
                s = line.rstrip("\r\n")
                if s == "":
                    continue
                if not s.startswith("data: "):
                    result["valid"] = False
                    result["error"] = "invalid SSE line"
                    break
                payload = s[6:].strip()
                if payload == "[DONE]":
                    done_seen = True
                    break
                try:
                    obj = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    result["valid"] = False
                    result["error"] = "invalid JSON in SSE"
                    break
                if not isinstance(obj, dict) or "choices" not in obj or not obj["choices"]:
                    result["valid"] = False
                    result["error"] = "no choices in SSE"
                    break
                delta = obj["choices"][0].get("delta", {})
                if not isinstance(delta, dict):
                    delta = {}
                piece = delta.get("content")
                if not piece:
                    piece = delta.get("reasoning_content")
                if isinstance(piece, str) and piece != "":
                    if ttft_abs is None:
                        ttft_abs = time.time()
                    result["tokens"] += 1
                    collected.append(piece)
            result["done"] = done_seen
            result["content"] = "".join(collected)
            if not done_seen:
                result["error"] = result["error"] or "no [DONE]"
            if not result["content"].strip():
                result["error"] = result["error"] or "empty response"
            if not result["valid"]:
                result["error"] = result["error"] or "invalid SSE"
        conn.close()
    except socket.timeout:
        result["timeout"] = True
        result["error"] = "timeout"
    except (http.client.HTTPException, OSError, ValueError, json.JSONDecodeError) as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["elapsed"] = time.time() - start
        if ttft_abs is not None:
            result["ttft"] = ttft_abs - start
        results[idx] = result


def main():
    parser = argparse.ArgumentParser(description="Load test for lfm-server OpenAI-compatible API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--conns", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--nonstream", action="store_true")
    args = parser.parse_args()

    if args.conns < 1:
        print("conns must be >= 1", file=sys.stderr)
        sys.exit(2)

    scheme, host, port, path = parse_url(args.url)
    results = [None] * args.conns
    threads = []
    for i in range(args.conns):
        t = threading.Thread(target=worker, args=(i, args, scheme, host, port, path, results), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"{'#':>3} {'TTFT(ms)':>10} {'tok/s':>8} {'tokens':>7}  status")
    for r in results:
        ttft_ms = f"{r['ttft'] * 1000:.1f}" if r["ttft"] is not None else "-"
        toks = f"{r['tokens'] / r['elapsed']:.2f}" if r["elapsed"] > 0 else "-"
        status = "OK" if not r["error"] else r["error"]
        print(f"{r['id']:>3} {ttft_ms:>10} {toks:>8} {r['tokens']:>7}  {status}")

    total_tokens = sum(r["tokens"] for r in results)
    wall = max((r["elapsed"] for r in results), default=0.0)
    agg = total_tokens / wall if wall > 0 else 0.0
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    min_ttft = min(ttfts) if ttfts else None
    max_ttft = max(ttfts) if ttfts else None
    errors = sum(1 for r in results if r["error"])
    timeouts = sum(1 for r in results if r["timeout"])

    print()
    print(f"Total time: {wall:.2f}s")
    print(f"Aggregate tok/s: {agg:.2f}")
    if min_ttft is not None:
        print(f"Min TTFT: {min_ttft * 1000:.1f} ms")
    else:
        print("Min TTFT: -")
    if max_ttft is not None:
        print(f"Max TTFT: {max_ttft * 1000:.1f} ms")
    else:
        print("Max TTFT: -")
    print(f"Errors: {errors}, Timeouts: {timeouts}")

    if errors or timeouts:
        sys.exit(1)
    for r in results:
        if r["error"]:
            sys.exit(1)
        if not args.nonstream:
            if not r["valid"] or not r["done"] or not r["content"].strip():
                sys.exit(1)
        else:
            if not r["valid"] or not r["content"].strip():
                sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
