#!/bin/bash
#SBATCH --job-name=job_bayesian_cfg
#SBATCH --output=job_bayesian_cfg.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --qos=genai-mingjieliu
#SBATCH --account=genai-mingjieliu
#SBATCH --distribution=cyclic:cyclic
#SBATCH --gpus=1
#SBATCH --partition=hpg-turin

module load cuda conda 
conda activate molguidance
pwd; hostname; date


# Pick the property for this task
PROPERTY="mu"

# Root where your per-property model folders live
MODEL_ROOT="../../checkpoints/qme14s/mu/classifier_free_guidance/epoch=1598-step=874215.ckpt"

# Run the optimizer for this property
python bayesian_cfg.py \
  --property   "$PROPERTY" \
  --model-dir  "$MODEL_ROOT" \
  --out-prefix "pkl-files"
