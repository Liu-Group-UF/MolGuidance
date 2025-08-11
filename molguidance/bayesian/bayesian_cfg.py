#!/usr/bin/env python
import os
import re
import argparse
import pickle

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from molguidance.utils.propmolflow_util import (
    get_model_ckpts,
    load_model_from_ckpt,
    molecules_to_sdf,
    compute_mae_from_sdf,
)
from molguidance.utils.guidance_sampler import sample_molecules_with_guidance

# ─── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--property',   required=True, help='e.g. gap, homo, mu, etc.')
parser.add_argument('--model-dir',  required=True)
parser.add_argument('--out-prefix', default='pkl-files')
parser.add_argument('--dataset-name', default='qme14s')
args = parser.parse_args()

# ─── Fixed settings ─────────────────────────────────────────────────────────
method           = 'cfg'
task             = 'many2many'
n_mols           = 1000
n_timesteps      = 100
max_batch_size   = 1000
prob_emb         = 'concatenate_sum'

search_space     = [
    Real(1.0, 4.0, name='guide_w1'),
    Real(1.0, 4.0, name='guide_w2'),
]


dataset_name = args.dataset_name
property_name = args.property
if dataset_name == 'qm9':
    # ─── Hard-coded QM9 numpy files directory ────────────────────────────────────
    qm9_val_path = (
        "/home/c.zeng/blue_mingjieliu/c.zeng/research-projects/"
        "MolGen/PropMolFlow/v2-new-data-qm9-property-distribution/npy-files"
    )
    vals = np.load(f"{qm9_val_path}/train_half_sampled_values_{property_name}.npy")[:n_mols]
    no_atoms = list(
        np.load(f"{qm9_val_path}/train_half_sampled_no_atoms_{property_name}.npy")[:n_mols]
    )

    normalization_file_path = (
        '/blue/mingjieliu/c.zeng/research-projects/'
        'MolGen/PropMolFlow/v2-new-data-qm9-property-distribution/'
        'data/qm9/train_data_property_normalization.pt'
    )
elif dataset_name == 'qme14s':
    qme14s_val_path = (
        "/blue/mingjieliu/jiruijin/diffusion/FlowMol/sample_results/qme14s/in_dis_sampling"
    )
    vals = np.load(f"{qme14s_val_path}/train_half_sampled_values_{property_name}.npy")[:n_mols]
    no_atoms = list(
        np.load(f"{qme14s_val_path}/train_half_sampled_no_atoms_{property_name}.npy")[:n_mols]
    )
    normalization_file_path = ("/blue/mingjieliu/jiruijin/diffusion/FlowMol/data/qme14s/train_data_property_normalization.pt")

multiple_values_to_one_property = [float(v) for v in vals]


# ─── Pick the first checkpoint ────────────────────────────────────────────────
# ckpt_list = get_model_ckpts(f"{args.model_dir}/{property_name}", "epoch*.ckpt")
# model_file = ckpt_list[0]
model_file = args.model_dir
epoch = re.search(r'epoch=(\d+)', os.path.basename(model_file)).group(1)

# ─── Define 2-D BO objective ─────────────────────────────────────────────────
@use_named_args(search_space)
def objective(guide_w1, guide_w2):
    foldername = (
        f"{method}_{property_name}/"
        f"bayes_opt_w1_{guide_w1:.2f}_w2_{guide_w2:.2f}"
    )
    sdf_file = (
        f"{foldername}/{method}_{property_name}_"
        f"{n_mols}_{task}_epoch{epoch}.sdf"
    )

    if not os.path.exists(sdf_file):
        os.makedirs(foldername, exist_ok=True)
        model = load_model_from_ckpt(model_file, method=method)
        molecules = sample_molecules_with_guidance(
            model,
            n_mols=n_mols,
            multiple_values_to_one_property=multiple_values_to_one_property,
            number_of_atoms=no_atoms,
            property_name=property_name,
            max_batch_size=max_batch_size,
            properties_handle_method=prob_emb,
            n_timesteps=n_timesteps,
            normalization_file_path=normalization_file_path,
            guide_w1=guide_w1,
            guide_w2=guide_w2,
            dataset_name=dataset_name,
        )
        molecules_to_sdf(molecules, sdf_file)

    mae = compute_mae_from_sdf(
        sdf_file, property_name, multiple_values_to_one_property
    )
    print(
        f"{property_name}, epoch={epoch}: "
        f"w1={guide_w1:.2f}, w2={guide_w2:.2f}, MAE={mae:.4f}"
    )
    return mae

# ─── Run Bayesian optimization ────────────────────────────────────────────────
res = gp_minimize(
    func=objective,
    dimensions=search_space,
    acq_func='EI',
    n_calls=50,
    n_initial_points=10,
    random_state=42,
    verbose=True,
)

# ─── Collect and dump results ────────────────────────────────────────────────
results_dict = {
    'best_w1':  res.x[0],
    'best_w2':  res.x[1],
    'best_mae': float(res.fun),
    'guide_ws': np.array(res.x_iters),
    'maes':     res.func_vals,
}

os.makedirs(args.out_prefix, exist_ok=True)
with open(f"{args.out_prefix}/double_w_{method}_results_{property_name}.pkl", "wb") as pf:
    pickle.dump(results_dict, pf)
