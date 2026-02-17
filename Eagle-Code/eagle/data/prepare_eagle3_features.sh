#!/usr/bin/env bash
set -euo pipefail
OUTDIR=${1:-prepared_data}
python -m eagle.data.prepare_eagle3_data --outdir "$OUTDIR"
