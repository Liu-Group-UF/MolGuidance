## Finding Optimal Guidance Weights via Bayesian Optimization

This workflow uses the QME14S dataset as an example to identify optimal guidance weight combinations for different guidance methods through Bayesian optimization.

**Example workflow (using AG guidance method):**

1. Navigate to the **ag** directory and run `sbatch job_ag_bayesian.sh` to submit the optimization job (or execute the script commands directly if SLURM is unavailable).

2. The optimal guidance weights and corresponding property prediction MAE will be saved as **pkl-files**. Use `load_pickle.ipynb` to load and visualize the results.

3. Update `jnb_sample_ag.sh` with the optimal guidance weights, then submit the job to generate molecules. Sampled molecules will be saved as SDF files in the corresponding guidance method directory.

## Metrics Calculation

This section demonstrates how to evaluate various metrics for different guidance methods using the QME14S dataset.

### 1. Stability, Uniqueness, and RDKit Validity
Run `get_scores.ipynb` in each guidance method directory to calculate stability, uniqueness, and RDKit validity metrics.

### 2. Property Prediction MAE
In each guidance method directory, run `sbatch job_property_align/job_property_align.sh` to calculate property prediction MAE (or execute the script commands directly if SLURM is unavailable). Results will be saved in `result.pt`.

### 3. PoseBusters Score
Navigate to the **PoseBusters** directory and run `sbatch jnb.sh` to calculate PoseBuster scores (or execute the script commands directly if SLURM is unavailable). 

**Prerequisite:** Ensure you have sampled SDF files from all four guidance methods before running this analysis.

### 4. Element Entropy, Bond Entropy, and Scaffold Diversity
Run `get_diversity_result.ipynb` to calculate element entropy, bond entropy, and scaffold diversity metrics.

**Prerequisite:** Ensure you have sampled SDF files from all four guidance methods before running this analysis.

## Required Modifications

- **Dataset path configuration**: In `MolGuidance/molguidance/property_regressor/configs/test_qme14s.yaml`, update `processed_data_dir` to point to your processed QME14S dataset location (absolute path recommended).

- **Regressor path configuration**: In `MolGuidance/molguidance/utils/propmolflow_util.py` (line 314), update `regressor_folder` to point to your `property_regressor` folder location (absolute path recommended).

- **Checkpoint configuration** (QM9 dataset): For convenience when running the model across all 6 properties, create a symbolic link named `last.ckpt` in each property folder within `MolGuidance/molguidance/property_regressor/model_output_qm9`:
```bash
   ln -s gvp-regressor-epoch=358-val_loss=0.0061.ckpt last.ckpt
```
   This eliminates the need to modify file paths for each property. Alternatively, specify the exact checkpoint file (e.g., `gvp-regressor-epoch=358-val_loss=0.0061.ckpt` for the **alpha** property).
