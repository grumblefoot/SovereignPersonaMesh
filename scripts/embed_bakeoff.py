"""
Embedding bake-off + threshold calibration (decision 4 / decision 6, embeddings
plan phases 5-6).

Scores one or more embedding models on a labelled corpus of RP memories and
queries: recall@5, MRR, latency percentiles, and a calibrated per-model cosine
DISTANCE threshold (the retriever's `max_cosine_distance`) chosen to maximise F1
over relevant/irrelevant pairs. Results print as a table and save as JSON.

Usage (SPM services NOT needed; talks straight to the embedding endpoint):
    .venv/bin/python -m scripts.embed_bakeoff \
        --corpus tests/fixtures/embed_bakeoff_corpus.json \
        --model embed-gemma-300m-FLM [--model Qwen3-Embedding-0.6B-GGUF] \
        --url http://127.0.0.1:13305/v1/embeddings \
        --out docs/plans/bakeoff_results.json

The corpus format is JSON: {"memories": [{"id", "text"}], "queries":
[{"query", "relevant": [memory ids]}]}. Swap in a dump of real chat/lore rows
any time; the harness doesn't care where the corpus came from.
"""
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx


def cosine_distance(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 1.0
    return 1.0 - dot / (na * nb)


async def embed_all(url: str, model: str, texts: list[str], batch: int = 16):
    """Embed texts; returns (vectors, per-call latencies in ms)."""
    vecs, lats = [], []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            t0 = time.perf_counter()
            r = await client.post(url, json={"model": model, "input": chunk})
            r.raise_for_status()
            lats.append((time.perf_counter() - t0) * 1000)
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            vecs.extend(d["embedding"] for d in data)
    return vecs, lats


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def evaluate(model: str, mem_vecs, query_vecs, memories, queries, latencies):
    mem_ids = [m["id"] for m in memories]
    recalls, mrrs = [], []
    pos_d, neg_d = [], []
    for q, qv in zip(queries, query_vecs):
        dists = sorted(
            ((cosine_distance(qv, mv), mid) for mv, mid in zip(mem_vecs, mem_ids)),
            key=lambda t: t[0])
        relevant = set(q["relevant"])
        top5 = [mid for _, mid in dists[:5]]
        recalls.append(len(relevant & set(top5)) / max(1, len(relevant)))
        rank = next((i + 1 for i, (_, mid) in enumerate(dists) if mid in relevant), None)
        mrrs.append(1.0 / rank if rank else 0.0)
        for d, mid in dists:
            (pos_d if mid in relevant else neg_d).append(d)

    # Threshold calibration: the cosine-distance cut that maximises F1 on the
    # labelled pairs (retriever keeps rows with distance <= threshold).
    best = (0.0, 0.35)
    for cand in sorted(set(round(d, 3) for d in pos_d + neg_d)):
        tp = sum(1 for d in pos_d if d <= cand)
        fp = sum(1 for d in neg_d if d <= cand)
        fn = len(pos_d) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, cand)

    lat_sorted = sorted(latencies)
    return {
        "model": model,
        "dim": len(mem_vecs[0]) if mem_vecs else 0,
        "recall_at_5": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(mrrs), 4),
        "latency_ms_p50": round(pct(lat_sorted, 0.50), 1),
        "latency_ms_p95": round(pct(lat_sorted, 0.95), 1),
        "pos_distance_mean": round(statistics.mean(pos_d), 4) if pos_d else None,
        "neg_distance_mean": round(statistics.mean(neg_d), 4) if neg_d else None,
        "calibrated_max_cosine_distance": best[1],
        "calibrated_f1": round(best[0], 4),
        "current_default_threshold": 0.35,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:13305/v1/embeddings")
    ap.add_argument("--out", default="")
    # EmbeddingGemma-style task prompts; empty = raw text (what SPM ships today).
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    memories, queries = corpus["memories"], corpus["queries"]
    print(f"Corpus: {len(memories)} memories, {len(queries)} queries\n")

    results = []
    for model in args.model:
        print(f"== {model}: embedding corpus ...")
        try:
            mem_vecs, lat1 = await embed_all(
                args.url, model, [args.doc_prefix + m["text"] for m in memories])
            q_vecs, lat2 = await embed_all(
                args.url, model, [args.query_prefix + q["query"] for q in queries],
                batch=1)  # queries go one at a time, like live chat
        except Exception as e:
            print(f"   SKIPPED ({e})\n")
            continue
        res = evaluate(model, mem_vecs, q_vecs, memories, queries, lat2)
        res["query_prefix"] = args.query_prefix
        res["doc_prefix"] = args.doc_prefix
        res["corpus_embed_ms_total"] = round(sum(lat1), 1)
        results.append(res)
        for k, v in res.items():
            print(f"   {k}: {v}")
        print()

    if args.out and results:
        Path(args.out).write_text(json.dumps(
            {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": args.url,
             "corpus": args.corpus, "results": results}, indent=2))
        print(f"Saved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
