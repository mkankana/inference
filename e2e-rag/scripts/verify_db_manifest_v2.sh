#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a behavioral-equivalence manifest.
#
# Usage (from repo root):
#   bash scripts/verify_db_manifest_v2.sh MANIFEST [OVERLAP_THRESHOLD]
#
#   MANIFEST:          path to the reference manifest JSON (required).
#   OVERLAP_THRESHOLD: minimum mean top-K retrieval overlap (default: 0.90).
#
# Configuration: see config.template.sh (uses INFERENCE_DB and
# INFERENCE_RETRIEVER_MODEL).
# =============================================================================

set -e

if [[ -z "$1" ]]; then
    echo "ERROR: manifest path required" >&2
    echo "Usage: $0 MANIFEST [OVERLAP_THRESHOLD]" >&2
    exit 1
fi

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"

MANIFEST="$1"
OVERLAP_THRESHOLD="${2:-0.90}"

echo "=== Verifying DB against manifest ==="
echo "  DB:                 ${INFERENCE_DB}"
echo "  Manifest:           ${MANIFEST}"
echo "  Overlap threshold:  ${OVERLAP_THRESHOLD}"
echo ""

python3 -u db_manifest_v2.py verify \
    --db "${INFERENCE_DB}" \
    --manifest "${MANIFEST}" \
    --overlap-threshold "${OVERLAP_THRESHOLD}"
