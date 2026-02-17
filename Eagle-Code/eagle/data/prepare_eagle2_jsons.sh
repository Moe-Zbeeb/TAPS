#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${1:-prepared_data}

python -m eagle.data.prepare_eagle3_data --outdir "$OUTDIR" --sharegpt --mathinstruct

python -m eagle.data.limit_jsonl --input "$OUTDIR/sharegpt_vicuna_unfiltered_train.jsonl" \
  --output "$OUTDIR/sharegpt_vicuna_70k.jsonl" --limit 70000

python -m eagle.data.limit_jsonl --input "$OUTDIR/mathinstruct_train.jsonl" \
  --output "$OUTDIR/mathinstruct_70k.jsonl" --limit 70000

rm -f "$OUTDIR/sharegpt_vicuna_unfiltered_train.jsonl" "$OUTDIR/mathinstruct_train.jsonl"
