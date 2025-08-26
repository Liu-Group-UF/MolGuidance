# MolGuidance
1. This molguidance framework is built on top of our previous work **[PropMolFlow](https://github.com/Liu-Group-UF/PropMolFlow)**.
2. You can also clone this repoistory from branch from our **[PropMolFlow](https://github.com/Liu-Group-UF/PropMolFlow)** repoistory without the need to create an new environment if you installed propmolflow before.
3. What new here is that we implement different guidance methods and guidance format for conditonal flow matching framework. Moreover, we introudced a new dataset **[QMe14S](https://pubs.acs.org/doi/10.1021/acs.jpclett.5c00839)** beyond **QM9** to test the scalability of the our methods.


## Environment Setup
Run the following commands in your terminal to set up `molguidance` (We have tested it on **Nvidia L4 and Blackwell B200** GPUs): 
```python
conda install mamba # if you do not have mamba
mamba create -n molguidance python=3.12 nvidia/label/cuda-12.8.1::cuda-toolkit
mamba activate molguidance
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-cluster torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0%2Bcu128.html
pip install torch-geometric
mamba install -c dglteam/label/th24_cu124 dgl=2.4.0.th24.cu124
pip install networkx==3.4 
pip install pytorch-lightning 
pip install einops rdkit numpy wandb py3Dmol useful_rdkit_utils
pip install torchtyping
pip install ase
pip install scikit-optimize
pip install -e .
```
## Preprocessing for Dataset 
### QM9
#### Download
We provide a corrected version of the QM9 SDF file originally from [DeepChem](https://github.com/deepchem/deepchem), fixing issues such as **invalid bond orders** and **non-zero net charges** in approximately **30,000 molecules**.

To download the revised SDF file, run:
```bash
wget https://zenodo.org/records/15700961/files/all_fixed_gdb9.zip
unzip all_fixed_gdb9.zip
rm all_fixed_gdb9.zip
```
After downloading, move the **all_fixed_gdb9.sdf** file to the `data/qm9_raw/` directory:
```bash
mv all_fixed_gdb9.sdf data/qm9_raw/
```
The revised SDF file is also hosted at our HuggingFace [ColabFit rQM9](https://huggingface.co/datasets/colabfit/rQM9).

#### Generate Training, Validation, Testing Data
```bash
python process_qm9_cond.py --config=configs/qm9/alpha/qm9_vanilla.yaml
```

### QMe14S
#### Download
We provide a cleaned SDF file of QMe14S where bonds information are first guessed by [Open Babel](https://github.com/openbabel/openbabel) then fixed for some unreasonable molecules' bond information. 

To download the revised SDF file, run:
```bash
wget https://zenodo.org/records/16847162/files/cleaned_qme14s.sdf.zip
unzip cleaned_qme14s.sdf.zip
rm cleaned_qme14s.sdf.zip.zip
```
After downloading, move the **all_fixed_gdb9.sdf** file to the `data/qm9_raw/` directory:
```bash
mv cleaned_qme14s.sdf.zip data/qme14s_raw/
```
#### Generate Training, Validation, Testing Data
```bash
python process_qme14s_cond.py --config=configs/qme14s/mu/qme14s_vanilla.yaml
```

## Vanilla Model
### Training
Train a PropMolFlow model without guidance. 
#### QM9 (take property alpha as example)
```python
# training from scratch
python train.py --config=configs/qm9/alpha/qm9_vanilla.yaml

# continue training from checkpoints
python train.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```
#### QMe14S
```python
# training from scratch
python train.py --config=configs/qme14s/mu/qme14s_vanilla.yaml

# continue training from checkpoints
python train.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```

### Sampling
#### QM9 (take property alpha as example) 
```python
python sample_condition.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "$sampling_result/qm9/alpha/vallina.sdf" --analyze
```
#### QMe14S 
```python
python sample_condition.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --output_file "$sampling_result/qme14s/vallina.sdf" --analyze
```

## Classifer-Free Guidance 
### Training
#### QM9 (take property alpha as example)
```python
# training from scratch
python train_with_guidance.py --config=configs/qm9/alpha/qm9_cfg.yaml

# continue training from checkpoints
python train_with_guidance.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```
#### QMe14S
```python
# training from scratch
python train_with_guidance.py --config=configs/qme14s/mu/qme14s_cfg.yaml

# continue training from checkpoints
python train_with_guidance.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```

### Sampling with Different Guidance Format
#### guidance on probability with linear format
```python
# QM9 with guidance weight = 1.5 for alpha
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "linear" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qm9/probability_linear/cfg.sdf" --analyze

# QMe14S with guidance weight = 1.5
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "linear" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qme14sprobability_linear/probability_linear/cfg.sdf" --analyze
```

#### guidance on probability with log format
```python
# QM9 with guidance weight = 1.5 for alpha
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qm9/probability_log/cfg.sdf" --analyze

# QMe14S with guidance weight = 1.5
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qme14s/probability_log/cfg.sdf" --analyze
```
#### guidance on rate matrix with linear format
```python
# QM9 with guidance weight = 1.5 for alpha
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "linear" --where_to_apply_guide "rate_matrix" --output_file "$sampling_result/qm9/rate_matrix_linear/cfg.sdf" --analyze

# QMe14S with guidance weight = 1.5
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "linear" --where_to_apply_guide "rate_matrix" --output_file "$sampling_result/qme14s/rate_matrix_linear/cfg.sdf" --analyze
```
#### guidance on rate matrix with log format
```python
# QM9 with guidance weight = 1.5 for alpha
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "rate_matrix" --output_file "$sampling_result/qm9/rate_matrix_log/cfg.sdf" --analyze

# QMe14S with guidance weight = 1.5
python sample_cfg.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "rate_matrix" --output_file "$sampling_result/qme14s/rate_matrix_log/cfg.sdf" --analyze
```

## Autoguidance
We use vanilla model as good model, and then train a inferior model to guide the vanilla model towards the direction with good property alignment.
### Training
#### QM9 (take property alpha as example)
```python
# training the inferor model from scratch
python train.py --config=configs/qm9/alpha/qm9_inferior.yaml
```
#### QMe14S
```python
# training the inferor model from scratch
python train.py --config=configs/qme14s/mu/qme14s_inferior.yaml
```
### Sampling
#### QM9 (take property alpha as example) 
```python
python sample_ag.py --good_model_checkpoint "$MODEL_CHECKPOINT" --bad_model_checkpoint "$BAD_MODEL" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qm9/probability_log/ag.sdf" --analyze
```
#### QMe14S 
```python
python sample_ag.py --good_model_checkpoint "$MODEL_CHECKPOINT" --bad_model_checkpoint "$BAD_MODEL" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_c 1.5 --dfm_type "campbell_rate_matrix" --guidance_format "log" --where_to_apply_guide "probabilities" --output_file "$sampling_result/qme14s/probability_log/ag.sdf" --analyze
```

## Modelguidance 
### Training
#### QM9 (take property alpha as example)
```python
# training from scratch
python train_with_guidance.py --config=configs/qm9/alpha/qm9_mg.yaml

# continue training from checkpoints
python train_with_guidance.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```
#### QMe14S
```python
# training from scratch
python train_with_guidance.py --config=configs/qme14s/mu/qme14s_mg.yaml

# continue training from checkpoints
python train_with_guidance.py --resume=checkpoints//alpha/epoch=1845-step=721785.ckpt
```
### Sampling
#### QM9 (take property alpha as example) 
```python
python sample_mg.py --model_checkpoint "$BAD_MODEL" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --multilple_values_file "sampling_input/qm9/in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "sampling_input/qm9/in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --guide_wight 1.5 --output_file "$sampling_result/qm9/probability_log/mg.sdf" --analyze
```
#### QMe14S 
```python
python sample_mg.py --model_checkpoint "$BAD_MODEL" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --multilple_values_file "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "sampling_input/qme14s/in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --normalization_file_path "data/qme14s/train_data_property_normalization.pt" --guide_wight 1.5 --output_file "$sampling_result/qme14s/probability_log/mg.sdf" --analyze
```

## Bonous: Classifier Guidance 


## Acknowledgements

Molguidance builds upon the source code from the following projects:
* [FlowMol](https://github.com/Dunni3/FlowMol)
* [Mattergen](https://github.com/microsoft/mattergen)
* [EDM2](https://github.com/NVlabs/edm2)
* [Diffusion-wo-CFG](https://github.com/tzco/Diffusion-wo-CFG)
* [Discrete_Guidance](https://github.com/hnisonoff/discrete_guidance)

