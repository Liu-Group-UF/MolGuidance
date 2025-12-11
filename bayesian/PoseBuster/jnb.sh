#!/bin/bash
#SBATCH --job-name=jnb      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=1                  # Number of MPI ranks
#SBATCH --cpus-per-task=10            # Number of cores per MPI rank
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1       # How many tasks on each node
#SBATCH --ntasks-per-socket=1        # How many tasks on each CPU or socket
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically on nodes and sockets
#SBATCH --mem-per-cpu=10gb          # Memory per processor
#SBATCH --time=5:00:00              # Time limit hrs:min:sec
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --qos=genai-mingjieliu
#SBATCH --account=genai-mingjieliu
pwd; hostname; date

ml conda cuda
conda activate molguidance
python get_pbscore.py
