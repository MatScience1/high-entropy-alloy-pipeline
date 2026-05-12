#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# submit_Tm_array.sh — SLURM Job Array for Tm Phase Coexistence
# 15 runs : 5 compositions × 3 T_guess values
# Submit  : sbatch slurm/submit_Tm_array.sh
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=Tm_coex
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --array=0-14%20
#SBATCH --output=logs/Tm_%A_%a.out
#SBATCH --error=logs/Tm_%A_%a.err

# ── dispatch table ────────────────────────────────────────────────────────
declare -A JOB_PATHS
declare -A JOB_LABELS
JOB_PATHS[0]="/home/users/razikaxz/pipeline1/runs/comp_000/tm_coexistence/T_3326"
JOB_PATHS[1]="/home/users/razikaxz/pipeline1/runs/comp_000/tm_coexistence/T_3695"
JOB_PATHS[2]="/home/users/razikaxz/pipeline1/runs/comp_000/tm_coexistence/T_4065"
JOB_PATHS[3]="/home/users/razikaxz/pipeline1/runs/comp_001/tm_coexistence/T_2606"
JOB_PATHS[4]="/home/users/razikaxz/pipeline1/runs/comp_001/tm_coexistence/T_2896"
JOB_PATHS[5]="/home/users/razikaxz/pipeline1/runs/comp_001/tm_coexistence/T_3186"
JOB_PATHS[6]="/home/users/razikaxz/pipeline1/runs/comp_002/tm_coexistence/T_2493"
JOB_PATHS[7]="/home/users/razikaxz/pipeline1/runs/comp_002/tm_coexistence/T_2770"
JOB_PATHS[8]="/home/users/razikaxz/pipeline1/runs/comp_002/tm_coexistence/T_3047"
JOB_PATHS[9]="/home/users/razikaxz/pipeline1/runs/comp_003/tm_coexistence/T_1915"
JOB_PATHS[10]="/home/users/razikaxz/pipeline1/runs/comp_003/tm_coexistence/T_2128"
JOB_PATHS[11]="/home/users/razikaxz/pipeline1/runs/comp_003/tm_coexistence/T_2341"
JOB_PATHS[12]="/home/users/razikaxz/pipeline1/runs/comp_004/tm_coexistence/T_1747"
JOB_PATHS[13]="/home/users/razikaxz/pipeline1/runs/comp_004/tm_coexistence/T_1941"
JOB_PATHS[14]="/home/users/razikaxz/pipeline1/runs/comp_004/tm_coexistence/T_2135"
JOB_LABELS[0]="comp_000_T3326"
JOB_LABELS[1]="comp_000_T3695"
JOB_LABELS[2]="comp_000_T4065"
JOB_LABELS[3]="comp_001_T2606"
JOB_LABELS[4]="comp_001_T2896"
JOB_LABELS[5]="comp_001_T3186"
JOB_LABELS[6]="comp_002_T2493"
JOB_LABELS[7]="comp_002_T2770"
JOB_LABELS[8]="comp_002_T3047"
JOB_LABELS[9]="comp_003_T1915"
JOB_LABELS[10]="comp_003_T2128"
JOB_LABELS[11]="comp_003_T2341"
JOB_LABELS[12]="comp_004_T1747"
JOB_LABELS[13]="comp_004_T1941"
JOB_LABELS[14]="comp_004_T2135"

# ── environment ───────────────────────────────────────────────────────────
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate grace

export GRACE_MODEL_DIR="$HOME/.cache/grace/GRACE-2L-OMAT"
export CUDA_VISIBLE_DEVICES="-1"
export OMP_NUM_THREADS=16
export TF_NUM_INTRAOP_THREADS=16
export TF_NUM_INTEROP_THREADS=2
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

# ── run ───────────────────────────────────────────────────────────────────
IDX=$SLURM_ARRAY_TASK_ID
RUN_DIR="${JOB_PATHS[$IDX]}"
LABEL="${JOB_LABELS[$IDX]}"

echo "[$IDX] $LABEL  host=$(hostname)  start=$(date)"

[ -d "$RUN_DIR" ]             || { echo "[$IDX] ERROR: $RUN_DIR not found"; exit 1; }
[ -f "$RUN_DIR/coexistence.in" ] || { echo "[$IDX] ERROR: coexistence.in missing"; exit 1; }

cd "$RUN_DIR"
mkdir -p dump

srun /home/users/razikaxz/lammps/build/lmp -in coexistence.in -log log.coexistence
EXIT_CODE=$?

echo "[$IDX] $LABEL  exit=$EXIT_CODE  end=$(date)"
exit $EXIT_CODE
