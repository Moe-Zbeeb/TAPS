#!/bin/bash
#
# Run Multi-Head Entropy Analysis
#
# This script runs the entropy analysis on multiple benchmarks,
# comparing the general and math EA heads.
#
# Usage:
#   ./scripts/run_entropy_analysis.sh [--max-questions N] [--benchmarks BENCH1 BENCH2 ...]

set -e

# Default paths
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to your base model (e.g. /path/to/Meta-Llama-3-8B-Instruct)}"
HEAD_GENERAL="${HEAD_GENERAL:-checkpoints/ShareGPT_20epochs}"
HEAD_MATH="${HEAD_MATH:-checkpoints/MathInstruct_20epochs/state_final}"
OUTPUT_DIR="${OUTPUT_DIR:-entropy_analysis_results}"

# Default benchmarks
BENCHMARKS="${BENCHMARKS:-mt_bench gsm8k math_500 svamp}"

# Optional max questions (for testing)
MAX_QUESTIONS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-questions)
            MAX_QUESTIONS="--max-questions $2"
            shift 2
            ;;
        --benchmarks)
            shift
            BENCHMARKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                BENCHMARKS="$BENCHMARKS $1"
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --head-general)
            HEAD_GENERAL="$2"
            shift 2
            ;;
        --head-math)
            HEAD_MATH="$2"
            shift 2
            ;;
        --skip-visualizations)
            SKIP_VIS="--skip-visualizations"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-questions N       Max questions per benchmark (for testing)"
            echo "  --benchmarks B1 B2 ...  Benchmarks to evaluate (default: mt_bench gsm8k math_500 svamp)"
            echo "  --output-dir DIR        Output directory (default: entropy_analysis_results)"
            echo "  --base-model PATH       Path to base LLaMA model"
            echo "  --head-general PATH     Path to general EA head checkpoint"
            echo "  --head-math PATH        Path to math EA head checkpoint"
            echo "  --skip-visualizations   Skip generating plots"
            echo ""
            echo "Environment variables:"
            echo "  BASE_MODEL_PATH         Default base model path"
            echo "  HEAD_GENERAL            Default general head path"
            echo "  HEAD_MATH               Default math head path"
            echo "  OUTPUT_DIR              Default output directory"
            echo "  BENCHMARKS              Default benchmarks (space-separated)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Multi-Head Entropy Analysis"
echo "=============================================="
echo "Base model:    $BASE_MODEL_PATH"
echo "General head:  $HEAD_GENERAL"
echo "Math head:     $HEAD_MATH"
echo "Benchmarks:    $BENCHMARKS"
echo "Output dir:    $OUTPUT_DIR"
echo "=============================================="

# Run the analysis
python "$REPO_ROOT/evaluation/entropy_analysis.py" \
    --base-model-path "$BASE_MODEL_PATH" \
    --head-general "$HEAD_GENERAL" \
    --head-math "$HEAD_MATH" \
    --benchmarks $BENCHMARKS \
    --output-dir "$OUTPUT_DIR" \
    $MAX_QUESTIONS \
    $SKIP_VIS

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
