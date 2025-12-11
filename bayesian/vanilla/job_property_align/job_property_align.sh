#!/bin/bash

#SBATCH --job-name=property_align
#SBATCH --output=jnb_property_align.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mail@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
##SBATCH --partition=hpg-dev
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=00:10:00


module load cuda conda 
conda activate molguidance
pwd; hostname; date

# change path to your MolGuidance directory
cd /blue/mingjieliu/jiruijin/github/MolGuidance

python molguidance/property_regressor/test_regressor.py \
 --checkpoint=molguidance/property_regressor/model_output_qme14s/mu/checkpoints/gvp-regressor-epoch=443-val_loss=0.0334.ckpt \
 --config=molguidance/property_regressor/configs/test_qme14s.yaml \
 --input=bayesian/vanilla/vanilla.sdf \
 --output=bayesian/vanilla/job_property_align/result.pt \
 --properties_values=sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy \
 --property_name=mu
