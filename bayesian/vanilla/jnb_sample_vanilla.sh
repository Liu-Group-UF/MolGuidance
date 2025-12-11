#!/bin/bash

#SBATCH --job-name=sample_vanilla
#SBATCH --output=jnb_sample_vanilla.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mail@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
##SBATCH --partition=hpg-dev
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=10gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# cd path to your MolGuidance repo
cd /blue/mingjieliu/jiruijin/github/MolGuidance/

MODEL_CHECKPOINT="checkpoints/qme14s/mu/vanilla/epoch=1621-step=887232.ckpt"
OUT_DIR="bayesian/vanilla"

module load cuda conda 
conda activate molguidance
pwd; hostname; date

python sample_condition.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --n_mols 10000 \
  --max_batch_size 128 \
  --n_timesteps 100 \
  --properties_handle_method "concatenate_sum" \
  --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" \
  --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" \
  --normalization_file_path "data/qme14s/train_data_property_normalization.pt" \
  --output_file "$OUT_DIR/vanilla.sdf" \
  --analyze
