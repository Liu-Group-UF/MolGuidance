#!/bin/bash

#SBATCH --job-name=alpha
#SBATCH --output=job_alpha.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mail@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
##SBATCH --partition=hpg-dev
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=10gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=1-00:00:00


module load cuda/12.4.1
pwd; hostname; date

# Activate the molguidance environment
source /blue/mingjieliu/jiruijin/program/anaconda3/bin/activate
conda activate molguidance

python train_classifier.py --config=configs/GVPRegressor/regressor_alpha.yaml --flowmol_config=configs/FlowMol/flowmol.yaml 