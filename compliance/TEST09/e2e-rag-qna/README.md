# TEST09 Compliance for E2E-RAG-QnA Workload

## Overview

TEST09 verifies that the output token length during performance runs matches expected values to prevent output truncation cheating.

## Thresholds

From `audit.config`: the reference mean output length of the final
`answer_generator` call is 273.81 tokens (`usage.completion_tokens` over the
reference `logs_result.json`, n=810), with a ±10% bound:

- Min output tokens: 246.43
- Max output tokens: 301.19

## Usage

Run from `e2e-rag/` in the inference repository. Settings come from
`config.default.sh` (override in `config.sh` or with exported variables).

### Automated workflow (recommended)

```bash
bash scripts/run_compliance_test09.sh
```

The script copies `audit.config`, runs the performance test with the perf cache
(`PERF_CACHE_FILE`, default `logs_result.json`; the run stops if it is
missing), runs `run_verification.py`, copies the results to the submission
directory and removes the copied `audit.config`.

### Manual workflow

```bash
cp ../compliance/TEST09/e2e-rag-qna/audit.config ./
python3 main_qna.py \
    --dataset_path frames-benchmark-dataset/frames_dataset.tsv \
    --database vector_html_hnsw_len768_ov32_word.db \
    --scenario Offline \
    --audit_conf audit.config \
    --log_dir run_output_test09 \
    --perf-test-mode logs_result.json
rm audit.config   # avoid running in compliance mode by accident

python3 ../compliance/TEST09/run_verification.py \
    -c run_output_test09 \
    -o submission/compliance/e2e-rag-qna/Offline \
    --audit-config ../compliance/TEST09/e2e-rag-qna/audit.config
```

### Expected result

`run_verification.py` reports the mean output tokens of the 824 samples and
`Overall: TEST PASS` when it is between 246.43 and 301.19.

## Notes

- **Component Measured:** `answer_generator` - this is the only component that LoadGen logs (the final answer generation step)
- **Dataset:** all 824 questions of `frames_dataset.tsv`, each issued once (`performance_issue_unique = 1`)
- **Workload:** Multi-hop RAG with iterative retrieval (max 5 iterations)
- The thresholds are based on the answer_generator component only, not the intermediate multi-shot retrieval steps
- Other components (check_sufficiency, generate_search_queries, evaluate_document_relevance) are internal to the SUT and not measured by TEST09
