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
pip install pytorch-lightning 
pip install einops rdkit numpy wandb py3Dmol useful_rdkit_utils
pip install torchtyping
pip install ase
pip install scikit-optimize
pip install -e .
```


