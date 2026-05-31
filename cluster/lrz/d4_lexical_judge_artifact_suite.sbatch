#!/bin/bash
#SBATCH --job-name=d4-lexical-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_lexical_judge_artifact_suite_v1}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export WORKDIR ARTROOT OUT_DIR

bash cluster/local/run_d4_lexical_judge_artifact_suite.sh
