# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


#!/usr/bin/env python3
"""Behavioral-equivalence vector DB check.

A DB built independently is accepted as equivalent when it uses the same
embedding model, the same corpus/chunking/parsing, and the same index
configuration -- even if the passages are stored in a different order and the
embeddings are numerically different. It does not require a byte-identical DB.

Workflow:
    # System A (after building DB):
    python3 db_manifest_v2.py write \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --output manifest_intel.json

    # System B (after building DB independently):
    python3 db_manifest_v2.py verify \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --manifest manifest_intel.json

Checks: passage count, embedding dimension, FAISS index parameters, an
order-independent corpus-set fingerprint, and top-K retrieval overlap against
reference queries within a tolerance.
"""

import argparse
import hashlib
import json
import sys
from typing import Dict, List

from db_manifest import (
    _gather_top_k,
    _load_db,
    _load_probe_queries,
    _open_manifest,
)

NUM_REFERENCE_QUERIES = 50
PROBE_TOP_K = 10
DEFAULT_OVERLAP_THRESHOLD = 0.90
DEFAULT_MODEL = "intfloat_e5-base-v2/e5-base-v2"


def _resolve_model(args, manifest=None):
    model = getattr(args, "retriever_model", None) or getattr(args, "embedding_model", None)
    if model:
        return model
    if manifest is not None:
        model = manifest.get("retriever_model") or manifest.get("embedding_model")
        if model:
            return model
    return DEFAULT_MODEL


def _add_model_args(parser):
    """--retriever_model / --embedding_model as interchangeable aliases."""
    parser.add_argument("--retriever_model", "--embedding_model", dest="retriever_model",
                        default=None)


def _corpus_set_sha256(db: "object") -> str:
    """SHA256 over the sorted set of per-passage text hashes.

    Order-independent (reordering passages yields the same value) but sensitive
    to chunking/parsing: any changed passage text changes the fingerprint. Text
    is hashed raw so whitespace differences count as real differences.
    """
    per_passage = []
    n = len(db._vector_store.index_to_docstore_id)
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        per_passage.append(
            hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest()
        )
    h = hashlib.sha256()
    for ph in sorted(per_passage):
        h.update(ph.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def _passage_hash_set(db: "object") -> set:
    """Set of per-passage raw-text SHA256 hashes."""
    out = set()
    n = len(db._vector_store.index_to_docstore_id)
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        out.add(hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest())
    return out


def _index_params(db: "object") -> Dict:
    """Index type / metric / HNSW build parameters."""
    index = db._vector_store.index
    try:
        import faiss
        base = faiss.downcast_index(index) if hasattr(faiss, "downcast_index") else index
    except Exception:
        base = index

    params = {
        "class": type(base).__name__,
        "dim": int(getattr(base, "d", 0)),
        "metric_type": int(getattr(base, "metric_type", -1)),
    }
    hnsw = getattr(base, "hnsw", None)
    if hnsw is not None:
        params["efConstruction"] = int(hnsw.efConstruction)
        params["efSearch"] = int(hnsw.efSearch)
        try:
            # HNSW stores up to 2*M neighbors at level 0.
            params["M"] = int(hnsw.nb_neighbors(0)) // 2
        except Exception:
            pass
    return params


def _norm_url(u: str) -> str:
    """Normalize a Wikipedia URL to its article key (scheme, .html, anchors and
    path separators dropped) so metadata-format differences don't mismatch."""
    u = u.lower()
    for pre in ("https://", "http://"):
        if u.startswith(pre):
            u = u[len(pre):]
    if u.endswith(".html"):
        u = u[:-5]
    u = u.replace("en.wikipedia.org/wiki/", "").replace("en.wikipedia.org_wiki_", "")
    u = u.split("#")[0]
    return u.replace("/", "_").strip("_")


def _overlap_vs_reference(cand_top: List[Dict], ref_top_map: Dict[int, List[str]],
                          top_k: int):
    """Return (mean_overlap, top1_rate, n). Overlap = fraction of the reference
    top-K URLs also present in the candidate top-K (order-independent)."""
    overlaps, top1, n = [], 0, 0
    for entry in cand_top:
        ref_urls = ref_top_map.get(entry["index"])
        if ref_urls is None:
            continue
        n += 1
        cand_urls = [_norm_url(u) for u in entry["top_k_urls"][:top_k]]
        ref_urls = [_norm_url(u) for u in ref_urls[:top_k]]
        sr, sc = set(ref_urls), set(cand_urls)
        overlaps.append(len(sr & sc) / (len(sr) or 1))
        if ref_urls and cand_urls and ref_urls[0] == cand_urls[0]:
            top1 += 1
    mean_ov = sum(overlaps) / (len(overlaps) or 1)
    return mean_ov, (top1 / n if n else 0.0), n


def cmd_write(args):
    model = _resolve_model(args)
    db = _load_db(args.db, model)
    total_passages = len(db._vector_store.index_to_docstore_id)

    print(f"[manifest] DB has {total_passages} passages, dim={db._embedding_dimension}")

    reference_queries = _load_probe_queries(args.dataset, NUM_REFERENCE_QUERIES)
    reference_top_k = _gather_top_k(db, reference_queries, PROBE_TOP_K)

    manifest = {
        "version": 2,
        "corpus_set_sha256": _corpus_set_sha256(db),
        # Write both names so the manifest is portable across repos that use
        # either "retriever_model" or "embedding_model".
        "retriever_model": model,
        "embedding_model": model,
        "index_params": _index_params(db),
        "total_passages": total_passages,
        "embedding_dim": db._embedding_dimension,
        "probe_top_k": PROBE_TOP_K,
        "reference_queries": reference_queries,
        "reference_top_k": reference_top_k,
    }

    with _open_manifest(args.output, "wt") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {args.output}")


def cmd_verify(args):
    with _open_manifest(args.manifest, "rt") as f:
        manifest = json.load(f)

    db = _load_db(args.db, _resolve_model(args, manifest))
    total_passages = len(db._vector_store.index_to_docstore_id)

    failures = []

    # Exact-match fields.
    if total_passages != manifest["total_passages"]:
        failures.append(
            f"total_passages mismatch: local={total_passages} manifest={manifest['total_passages']}"
        )
    if db._embedding_dimension != manifest["embedding_dim"]:
        failures.append(
            f"embedding_dim mismatch: local={db._embedding_dimension} "
            f"manifest={manifest['embedding_dim']}"
        )

    # Index parameters / algorithm.
    local_params = _index_params(db)
    if local_params != manifest["index_params"]:
        failures.append(
            f"index_params mismatch:\n"
            f"  local    = {local_params}\n"
            f"  manifest = {manifest['index_params']}"
        )
    print(f"[verify] index params: {local_params}")

    # Corpus set fingerprint (order-independent).
    local_set_sha = _corpus_set_sha256(db)
    if local_set_sha == manifest["corpus_set_sha256"]:
        print("[verify] corpus set: match")
    else:
        n_distinct = len(_passage_hash_set(db))
        failures.append(
            f"corpus set sha256 mismatch (chunking/parsing/corpus differs):\n"
            f"  local    = {local_set_sha} ({n_distinct} distinct passages)\n"
            f"  manifest = {manifest['corpus_set_sha256']}"
        )

    # Top-K retrieval overlap vs reference queries.
    cand_top = _gather_top_k(db, manifest["reference_queries"], manifest["probe_top_k"])
    ref_top = {r["index"]: r["top_k_urls"] for r in manifest["reference_top_k"]}
    mean_ov, top1, nq = _overlap_vs_reference(cand_top, ref_top, manifest["probe_top_k"])
    print(f"[verify] retrieval: {nq} queries, top-{manifest['probe_top_k']} "
          f"mean overlap={mean_ov:.3f} (threshold={args.overlap_threshold}) top-1={top1:.3f}")
    if mean_ov < args.overlap_threshold:
        failures.append(
            f"retrieval overlap below threshold: "
            f"mean={mean_ov:.3f} < threshold={args.overlap_threshold}"
        )

    if failures:
        print("\n[verify] FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[verify] OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("write", help="Generate a reference manifest from a DB.")
    pw.add_argument("--db", required=True)
    _add_model_args(pw)
    pw.add_argument("--dataset", default="data/frames_dataset.tsv")
    pw.add_argument("--output", required=True)
    pw.set_defaults(func=cmd_write)

    pv = sub.add_parser("verify", help="Verify a DB against a reference manifest.")
    pv.add_argument("--db", required=True)
    pv.add_argument("--manifest", required=True)
    _add_model_args(pv)
    pv.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    pv.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
