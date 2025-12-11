#!/bin/bash
#SBATCH --job-name=job_ag_bayesian
#SBATCH --output=job_ag_bayesian.out
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --qos=genai-mingjieliu
#SBATCH --account=genai-mingjieliu
#SBATCH --distribution=cyclic:cyclic
#SBATCH --gpus=1
#SBATCH --partition=hpg-turin

module load cuda conda 
conda activate molguidance
pwd; hostname; date

python bayesian_ag.py \
     --property_name          "mu" \
     --main_model_folder      "../../checkpoints/qme14s/mu/vanilla/epoch=1621-step=887232.ckpt" \
     --weak_model_folder      "../../checkpoints/qme14s/mu/auto_guidance/model-step=step=40000.ckpt" \
     --qm9_val_path           "../../sampling_input/qme14s/in_distribution_sampling" \
     --normalization_file_path "../../data/qme14s/train_data_property_normalization.pt" \
     --prob_emb               "concatenate_sum" \
     --method                 "ag" \
     --n_mols                 1000 \
     --n_calls                50 \
     --n_initial              10 \
     --random_seed            42 \
     --task                   "many2many" # many2many means in distribution sampling

pwd; hostname; date