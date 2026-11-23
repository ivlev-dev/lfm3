#!/usr/bin/env python3
"""A/B driver for the myelin quantization quality stand.

Runs the ``quality_eval`` worker once per weight format (the format is chosen by
``MYELIN_CPU_*`` env variables and is read once at model load, so every format
needs its own process), parses its JSONL output and prints:

* a per-format summary (perplexity, degeneration rate, repetition, confidence);
* a format x prompt verdict grid;
* Markdown/CSV artefacts under ``--out-dir``.

Only the standard library is used. See ``README.md`` for the metric and
acceptance-threshold definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
DEFAULT_BIN = REPO_ROOT / "myelin" / "target" / "release" / "examples" / "quality_eval"

#: Formats under comparison. ``env`` is merged into the worker environment.
FORMATS = [
    {
        "id": "ref",
        "label": "Q4_K native (ref)",
        "note": "no requantisation; the shipped checkpoint values",
        "env": {"MYELIN_CPU_Q4K": "1"},
    },
    {
        "id": "perrow",
        "label": "int4 per-row",
        "note": "absmax/7 per row (current DSP format)",
        "env": {"MYELIN_CPU_INT4": "1"},
    },
    {
        "id": "blk256",
        "label": "int4 blk256",
        "note": "symmetric absmax per 256 inputs",
        "env": {"MYELIN_CPU_WBLOCK": "256"},
    },
    {
        "id": "blk128",
        "label": "int4 blk128",
        "note": "symmetric absmax per 128 inputs",
        "env": {"MYELIN_CPU_WBLOCK": "128"},
    },
    {
        "id": "perrow-h32",
        "label": "int4 per-row + H32",
        "note": "per-row scale after block-Hadamard rotation",
        "env": {"MYELIN_CPU_INT4": "1", "MYELIN_CPU_WHAD": "32"},
    },
    {
        "id": "blk256-h32-conv16",
        "label": "blk256 + H32 + CONV16",
        "note": "candidate DSP combination",
        "env": {
            "MYELIN_CPU_WBLOCK": "256",
            "MYELIN_CPU_WHAD": "32",
            "MYELIN_CPU_CONV16": "1",
        },
    },
]

#: Acceptance thresholds. A format is accepted when every budget is met.
THRESHOLDS = {
    "ppl_ratio_max": 1.05,
    "ppl_text_ratio_max": 1.10,
    "degen_rate_max": 0.15,
    "cycle_rate_max": 0.10,
    "repeat4_mean_max": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="quality_eval binary")
    parser.add_argument("--prompts", type=Path, default=EVAL_DIR / "prompts.jsonl")
    parser.add_argument("--corpus", type=Path, default=EVAL_DIR / "corpus.jsonl")
    parser.add_argument("--max-tokens", type=int, default=40, help="greedy tokens per prompt")
    parser.add_argument("--jobs", type=int, default=2, help="formats evaluated concurrently")
    parser.add_argument(
        "--pin",
        action="store_true",
        help="pin every worker to its own CPU subset with taskset (recommended with --jobs > 1)",
    )
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR / "results")
    parser.add_argument(
        "--formats",
        default=",".join(item["id"] for item in FORMATS),
        help="comma separated format ids (default: all)",
    )
    parser.add_argument("--reference", default="ref", help="format id used as the baseline")
    parser.add_argument("--no-ppl", action="store_true", help="skip perplexity")
    parser.add_argument("--build", action="store_true", help="cargo build the worker first")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="re-render from <out-dir>/<id>.jsonl instead of running the worker",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a format misses an acceptance threshold",
    )
    return parser.parse_args()


def build_worker(binary: Path) -> None:
    print(f"building worker -> {binary}", flush=True)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "myelin-core",
            "--features",
            "hf-tokenizer",
            "--example",
            "quality_eval",
        ],
        cwd=REPO_ROOT / "myelin",
        check=True,
    )
    if not binary.exists():
        raise SystemExit(f"worker not found at {binary}")


def parse_output(fmt: dict, out_file: Path, log_file: Path, wall_seconds: float) -> dict:
    records = [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]
    meta = next(item for item in records if item["kind"] == "meta")
    prompts_records = [item for item in records if item["kind"] == "prompt"]
    ppl_records = [item for item in records if item["kind"] == "ppl"]
    summary = next(item for item in records if item["kind"] == "summary")

    wstats = None
    for line in log_file.read_text(errors="replace").splitlines():
        if line.startswith("wstats:"):
            wstats = line
    return {
        "format": fmt,
        "meta": meta,
        "prompts": prompts_records,
        "ppl": ppl_records,
        "summary": summary,
        "wall_seconds": wall_seconds,
        "wstats": wstats,
    }


def load_cached(fmt: dict, out_dir: Path) -> dict:
    out_file = out_dir / f"{fmt['id']}.jsonl"
    log_file = out_dir / f"{fmt['id']}.log"
    if not out_file.exists():
        raise SystemExit(f"no cached result for {fmt['id']} at {out_file}")
    return parse_output(fmt, out_file, log_file, 0.0)


def run_format(
    fmt: dict,
    binary: Path,
    prompts: Path,
    corpus: Path,
    max_tokens: int,
    out_dir: Path,
    no_ppl: bool,
    cpu_set: str | None = None,
) -> dict:
    out_file = out_dir / f"{fmt['id']}.jsonl"
    log_file = out_dir / f"{fmt['id']}.log"
    env = os.environ.copy()
    env.update({key: str(value) for key, value in fmt["env"].items()})
    # Weight cosine / relRMSE for the report; harmless extra stderr lines.
    env["MYELIN_CPU_WSTATS"] = "1"
    command = [
        str(binary),
        "--prompts",
        str(prompts),
        "--corpus",
        str(corpus),
        "--max-tokens",
        str(max_tokens),
        "--format",
        fmt["id"],
        "--out",
        str(out_file),
    ]
    if no_ppl:
        command.append("--no-ppl")
    if cpu_set:
        command = ["taskset", "-c", cpu_set, *command]

    started = time.time()
    with log_file.open("w") as log:
        completed = subprocess.run(command, env=env, stderr=log, stdout=subprocess.PIPE)
    elapsed = time.time() - started
    if completed.returncode != 0:
        tail = "\n".join(log_file.read_text(errors="replace").splitlines()[-15:])
        raise SystemExit(f"format {fmt['id']} failed (see {log_file}):\n{tail}")

    return parse_output(fmt, out_file, log_file, elapsed)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate(result: dict) -> dict:
    prompts = result["prompts"]
    metrics = [item["metrics"] for item in prompts]
    ppl = result["ppl"]

    def weighted_ppl(items: list[dict]) -> float:
        predicted = sum(item["predicted"] for item in items)
        if not predicted:
            return float("nan")
        return math.exp(sum(item["nll_sum"] for item in items) / predicted)

    top1 = (
        sum(item["top1_acc"] * item["predicted"] for item in ppl)
        / sum(item["predicted"] for item in ppl)
        if ppl
        else float("nan")
    )
    top5 = (
        sum(item["top5_acc"] * item["predicted"] for item in ppl)
        / sum(item["predicted"] for item in ppl)
        if ppl
        else float("nan")
    )
    english = [item for item in ppl if item["lang"] == "en"]
    runs = [item["max_token_run"] for item in metrics]
    cycles = [item for item in metrics if item["cycle_coverage"] >= 0.5]
    return {
        "ppl": weighted_ppl(ppl),
        "ppl_en": weighted_ppl(english),
        "nll_tokens": sum(item["predicted"] for item in ppl),
        "top1_acc": top1,
        "top5_acc": top5,
        "degen_rate": mean([1.0 if item["degenerate"] else 0.0 for item in prompts]),
        "cycle_rate": len(cycles) / len(prompts) if prompts else 0.0,
        "repeat4_mean": mean([item["repeat4"] for item in metrics]),
        "repeat2_mean": mean([item["repeat2"] for item in metrics]),
        "maxrun_mean": mean([float(value) for value in runs]),
        "maxrun_p95": percentile([float(value) for value in runs], 0.95),
        "maxfunc_p95": percentile([float(item["max_function_run"]) for item in metrics], 0.95),
        "function_frac_mean": mean([item["function_frac"] for item in metrics]),
        "top1_seen_mean": mean([item["top1_seen_frac"] for item in metrics]),
        "entropy_mean": mean([item["mean_entropy"] for item in metrics]),
        "top1_prob_mean": mean([item["mean_top1_prob"] for item in metrics]),
        "fact_rate": mean(
            [
                1.0 if item["fact_hit"] is True else 0.0
                for item in prompts
                if item["fact_hit"] is not None
            ]
        ),
        "fact_total": sum(1 for item in prompts if item["fact_hit"] is not None),
    }


def parse_wstats(line: str | None) -> tuple[str, float, float]:
    """Return (format, cosine, relrmse) from a wstats line."""
    if not line:
        return ("", float("nan"), float("nan"))
    fields = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    try:
        cosine = float(fields.get("cos", "nan"))
    except ValueError:
        cosine = float("nan")
    try:
        relrmse = float(fields.get("relrmse", "nan"))
    except ValueError:
        relrmse = float("nan")
    return (fields.get("format", ""), cosine, relrmse)


def fmt_float(value: float, digits: int = 3) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value != value:
        return "—"
    return f"{value * 100:.{digits}f}"


def render_summary(
    results: list[dict], reference: dict, reference_agg: dict, args
) -> str:
    lines: list[str] = []
    lines.append("# Quantization quality eval")
    lines.append("")
    lines.append(
        f"Model: `{reference['meta']['model']}`, greedy, "
        f"{reference['meta']['max_tokens']} tokens/prompt, seed {reference['meta']['seed']}."
    )
    lines.append(f"Prompts: `{args.prompts}`, perplexity corpus: `{args.corpus}`.")
    lines.append("")
    lines.append("| format | env | relRMSE | bytes/w | PPL | ΔPPL vs ref | Δ per text | top1 | degen% | cycle% | funcrun p95 | rep4 | H, nats | top1p | tok/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    reference_by_text = {item["id"]: item["ppl"] for item in reference["ppl"]}
    aggregates = {}
    for result in results:
        fmt = result["format"]
        agg = aggregate(result)
        aggregates[fmt["id"]] = agg
        _, _, relrmse = parse_wstats(result["wstats"])
        env = " ".join(f"{key}={value}" for key, value in fmt["env"].items())
        delta = agg["ppl"] / reference_agg["ppl"] - 1.0 if reference_agg["ppl"] else float("nan")
        text_ratios = [
            item["ppl"] / reference_by_text[item["id"]] - 1.0
            for item in result["ppl"]
            if item["id"] in reference_by_text and reference_by_text[item["id"]]
        ]
        delta_text = mean(text_ratios)
        lines.append(
            "| {label} | `{env}` | {relrmse} | {bytesw} | {ppl} | {delta} | {dtext} | {top1} | {degen} | "
            "{cycle} | {maxrun} | {rep4} | {entropy} | {top1p} | {speed} |".format(
                label=fmt["label"],
                env=env,
                relrmse=fmt_float(relrmse, 4),
                bytesw=fmt_float(parse_bytes_per_weight(result["wstats"]), 4),
                ppl=fmt_float(agg["ppl"]),
                delta=("—" if delta != delta else f"{delta * 100:+.1f}%"),
                dtext=("—" if delta_text != delta_text else f"{delta_text * 100:+.1f}%"),
                top1=fmt_pct(agg["top1_acc"]),
                degen=fmt_pct(agg["degen_rate"]),
                cycle=fmt_pct(agg["cycle_rate"]),
                maxrun=fmt_float(agg["maxfunc_p95"], 1),
                rep4=fmt_float(agg["repeat4_mean"]),
                entropy=fmt_float(agg["entropy_mean"], 2),
                top1p=fmt_pct(agg["top1_prob_mean"]),
                speed=fmt_float(result["summary"]["tok_per_second"], 2),
            )
        )

    lines.append("")
    lines.append("## Perplexity per corpus text")
    lines.append("")
    text_ids = [item["id"] for item in reference["ppl"]]
    header = "| text | " + " | ".join(result["format"]["label"] for result in results) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(results))
    lookup = {
        result["format"]["id"]: {item["id"]: item for item in result["ppl"]} for result in results
    }
    for text_id in text_ids:
        cells = []
        for result in results:
            item = lookup[result["format"]["id"]].get(text_id)
            cells.append(fmt_float(item["ppl"]) if item else "—")
        lines.append(f"| {text_id} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Prompt x format verdict")
    lines.append("")
    lines.append("`ok` = no degeneration flag, `DEG` = flagged, `—` = not run.")
    lines.append("")
    lines.append("| prompt | type | " + " | ".join(item["format"]["label"] for item in results) + " |")
    lines.append("|---|---|" + "---|" * len(results))
    prompt_ids = [item["id"] for item in reference["prompts"]]
    for prompt_id in prompt_ids:
        kind = next(
            (item["type"] for item in reference["prompts"] if item["id"] == prompt_id), ""
        )
        cells = []
        for result in results:
            item = next((entry for entry in result["prompts"] if entry["id"] == prompt_id), None)
            if item is None:
                cells.append("—")
            else:
                cells.append("DEG" if item["degenerate"] else "ok")
        lines.append(f"| {prompt_id} | {kind} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    lines.append(
        "Thresholds: PPL ratio vs ref ≤ {ppl_ratio_max}, degeneration rate ≤ {degen_rate_max}, "
        "cyclic-prompt rate ≤ {cycle_rate_max}, mean repeat4 ≤ {repeat4_mean_max}.".format(
            **THRESHOLDS
        )
    )
    lines.append("")
    lines.append("| format | PPL ratio | max per-text | degen | cycle | rep4 | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for result in results:
        fmt = result["format"]
        agg = aggregates[fmt["id"]]
        ratio = agg["ppl"] / reference_agg["ppl"] if reference_agg["ppl"] else float("nan")
        text_ratios = [
            item["ppl"] / reference_by_text[item["id"]]
            for item in result["ppl"]
            if item["id"] in reference_by_text and reference_by_text[item["id"]]
        ]
        max_text = max(text_ratios) if text_ratios else float("nan")
        failures = []
        if ratio != ratio or ratio > THRESHOLDS["ppl_ratio_max"]:
            failures.append("PPL")
        if max_text != max_text or max_text > THRESHOLDS["ppl_text_ratio_max"]:
            failures.append("per-text PPL")
        if agg["degen_rate"] > THRESHOLDS["degen_rate_max"]:
            failures.append("degeneration")
        if agg["cycle_rate"] > THRESHOLDS["cycle_rate_max"]:
            failures.append("cycles")
        if agg["repeat4_mean"] > THRESHOLDS["repeat4_mean_max"]:
            failures.append("repetition")
        verdict = "PASS" if not failures else "FAIL: " + ", ".join(failures)
        lines.append(
            f"| {fmt['label']} | {fmt_float(ratio, 3)} | {fmt_float(max_text, 3)} | "
            f"{fmt_pct(agg['degen_rate'])} | {fmt_pct(agg['cycle_rate'])} | "
            f"{fmt_float(agg['repeat4_mean'])} | {verdict} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_bytes_per_weight(line: str | None) -> float:
    if not line:
        return float("nan")
    for part in line.split():
        if part.startswith("bytes/weight="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return float("nan")
    return float("nan")


def write_artifacts(results: list[dict], out_dir: Path) -> None:
    rows = []
    for result in results:
        fmt = result["format"]
        for item in result["prompts"]:
            row = {
                "format": fmt["id"],
                "format_label": fmt["label"],
                "id": item["id"],
                "type": item["type"],
                "lang": item["lang"],
                "degenerate": int(bool(item["degenerate"])),
                "fact_hit": "" if item["fact_hit"] is None else int(bool(item["fact_hit"])),
                "text": item["text"],
                "stop": item["stop"],
            }
            row.update(item["metrics"])
            rows.append(row)
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "generations.md").open("w") as handle:
        handle.write("# Greedy generations\n\n")
        for result in results:
            handle.write(f"## {result['format']['label']}\n\n")
            for item in result["prompts"]:
                flag = "DEG" if item["degenerate"] else "ok"
                text = item["text"].replace("\n", "\\n")
                handle.write(f"- **{item['id']}** ({flag}): `{text}`\n")
            handle.write("\n")

    payload = {
        "formats": [
            {
                "id": result["format"]["id"],
                "label": result["format"]["label"],
                "env": result["format"]["env"],
                "wstats": result["wstats"],
                "summary": result["summary"],
                "aggregate": aggregate(result),
                "prompts": result["prompts"],
                "ppl": result["ppl"],
            }
            for result in results
        ]
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    if not args.from_cache and (args.build or not args.bin.exists()):
        build_worker(args.bin)
    if not args.prompts.exists() or not args.corpus.exists():
        raise SystemExit("prompts/corpus file is missing")

    wanted = [item.strip() for item in args.formats.split(",") if item.strip()]
    selected = [item for item in FORMATS if item["id"] in wanted]
    if not selected:
        raise SystemExit(f"no known format in {wanted}")
    if args.reference not in {item["id"] for item in selected}:
        raise SystemExit(f"reference format {args.reference!r} is not selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cpu_sets: list[str | None] = [None] * len(selected)
    if args.pin:
        if not shutil.which("taskset"):
            raise SystemExit("--pin needs taskset")
        cores = os.cpu_count() or 1
        for index in range(len(selected)):
            lane = index % args.jobs
            start = lane * cores // args.jobs
            end = (lane + 1) * cores // args.jobs - 1
            cpu_sets[index] = str(start) if start == end else f"{start}-{end}"
    print(
        f"evaluating {len(selected)} formats x {args.max_tokens} tokens, "
        f"{args.jobs} jobs, out {args.out_dir}",
        flush=True,
    )
    if args.pin:
        print(
            "pinned: "
            + ", ".join(
                f"{fmt['id']}->{cpu_set}" for fmt, cpu_set in zip(selected, cpu_sets)
            ),
            flush=True,
        )

    results: list[dict] = []
    if args.from_cache:
        results = [load_cached(fmt, args.out_dir) for fmt in selected]
        print(f"re-rendered {len(results)} cached formats from {args.out_dir}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    run_format,
                    fmt,
                    args.bin,
                    args.prompts,
                    args.corpus,
                    args.max_tokens,
                    args.out_dir,
                    args.no_ppl,
                    cpu_sets[index],
                ): fmt
                for index, fmt in enumerate(selected)
            }
            for future, fmt in futures.items():
                result = future.result()
                print(
                    f"  {fmt['id']:<22} done in {result['wall_seconds']:.0f}s",
                    flush=True,
                )
                results.append(result)

    results.sort(key=lambda item: wanted.index(item["format"]["id"]))
    reference = next(item for item in results if item["format"]["id"] == args.reference)
    reference_agg = aggregate(reference)

    table = render_summary(results, reference, reference_agg, args)
    summary_path = args.out_dir / "summary.md"
    summary_path.write_text(table)
    write_artifacts(results, args.out_dir)
    print()
    print(table)
    print(f"artefacts: {summary_path}, {args.out_dir / 'results.json'}, "
          f"{args.out_dir / 'metrics.csv'}, {args.out_dir / 'generations.md'}")

    if args.strict:
        reference_by_text = {item["id"]: item["ppl"] for item in reference["ppl"]}
        failed = []
        for result in results:
            agg = aggregate(result)
            ratio = agg["ppl"] / reference_agg["ppl"] if reference_agg["ppl"] else float("nan")
            text_ratios = [
                item["ppl"] / reference_by_text[item["id"]]
                for item in result["ppl"]
                if item["id"] in reference_by_text and reference_by_text[item["id"]]
            ]
            max_text = max(text_ratios) if text_ratios else float("nan")
            if (
                ratio != ratio
                or ratio > THRESHOLDS["ppl_ratio_max"]
                or max_text != max_text
                or max_text > THRESHOLDS["ppl_text_ratio_max"]
                or agg["degen_rate"] > THRESHOLDS["degen_rate_max"]
                or agg["cycle_rate"] > THRESHOLDS["cycle_rate_max"]
                or agg["repeat4_mean"] > THRESHOLDS["repeat4_mean_max"]
            ):
                failed.append(result["format"]["id"])
        if failed:
            print(f"strict mode: failing formats: {', '.join(failed)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
