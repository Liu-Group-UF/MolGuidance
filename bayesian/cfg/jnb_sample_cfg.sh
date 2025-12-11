#!/bin/bash

#SBATCH --job-name=sample_cfg
#SBATCH --output=jnb_sample_cfg.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mail@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=mingjieliu
#SBATCH --qos=mingjieliu
##SBATCH --partition=hpg-dev
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=10gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# cd path to your MolGuidance repo
cd /blue/mingjieliu/jiruijin/github/MolGuidance/

MODEL_CHECKPOINT="checkpoints/qme14s/mu/classifier_free_guidance/epoch=1598-step=874215.ckpt"
OUT_DIR="."

module load cuda conda 
conda activate molguidance
pwd; hostname; date

python sample_cfg.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --n_mols 10000 \
  --max_batch_size 128 \
  --n_timesteps 100 \
  --properties_handle_method "concatenate_sum" \
  --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" \
  --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" \
  --normalization_file_path "data/qme14s/train_data_property_normalization.pt" \
  --guide_w_x 3.13 \
  --guide_w_a 1.88 \
  --guide_w_c 1.88 \
  --guide_w_e 1.88 \
  --dfm_type "campbell" \
  --output_file "$OUT_DIR/cfg_guidance_best_w.sdf" \
  --analyze
