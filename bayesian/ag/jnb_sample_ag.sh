#!/bin/bash

#SBATCH --job-name=sample_ag
#SBATCH --output=jnb_sample_ag_1_0.out
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
#SBATCH --time=02:00:00

# cd path to your MolGuidance repo
cd /blue/mingjieliu/jiruijin/github/MolGuidance/

GOOD_MODEL="checkpoints/qme14s/mu/vanilla/epoch=1621-step=887232.ckpt"
BAD_MODEL="checkpoints/qme14s/mu/auto_guidance/model-step=step=40000.ckpt"
OUT_DIR="bayesian/ag"

module load cuda conda 
conda activate molguidance
pwd; hostname; date

python sample_ag.py \
  --good_model_checkpoint "$GOOD_MODEL" \
  --bad_model_checkpoint "$BAD_MODEL" \
  --n_mols 10000 \
  --max_batch_size 128 \
  --n_timesteps 100 \
  --properties_handle_method "concatenate_sum" \
  --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" \
  --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" \
  --normalization_file_path "data/qme14s/train_data_property_normalization.pt" \
  --guide_w_x 2.7 \
  --guide_w_a 1.1 \
  --guide_w_c 1.1 \
  --guide_w_e 1.1 \
  --dfm_type "campbell" \
  --output_file "$OUT_DIR/ag_guidance_best_w.sdf" \
  --analyze
