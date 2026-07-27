#!/bin/bash
# =============================================================================
# Generate a behavioral-equivalence DB manifest from this system's vector DB.
#
# Usage (from repo root):
#   bash scripts/write_db_manifest_v2.sh [OUTPUT]
#
#   OUTPUT: optional path for the manifest JSON. Use .json.gz to compress.
#           default: db_manifest_v2_$(hostname -s).json.gz
#
# Configuration: see config.template.sh (uses INFERENCE_DB and
# INFERENCE_RETRIEVER_MODEL).
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"
INFERENCE_RETRIEVER_MODEL="${INFERENCE_RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}"

OUTPUT="${1:-db_manifest_v2_$(hostname -s).json.gz}"

echo "=== Writing DB manifest ==="
echo "  DB:        ${INFERENCE_DB}"
echo "  Retriever: ${INFERENCE_RETRIEVER_MODEL}"
echo "  Output:    ${OUTPUT}"
echo ""

python3 -u db_manifest_v2.py write \
    --db "${INFERENCE_DB}" \
    --retriever_model "${INFERENCE_RETRIEVER_MODEL}" \
    --output "${OUTPUT}"
