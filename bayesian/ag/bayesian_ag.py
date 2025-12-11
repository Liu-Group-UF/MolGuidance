#!/usr/bin/env python
import argparse
import os
import pickle
import warnings

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from molguidance.utils.propmolflow_util import (
    molecules_to_sdf,
    get_model_ckpts,
    load_model_from_ckpt,
    compute_mae_from_sdf,
)
from molguidance.utils.autoguidance_sampler import sample_molecules_with_auto_guidance

warnings.filterwarnings("ignore")

# 1. Define a 2D search space
search_space     = [
    Real(1.0, 4.3, name='guide_w1'),
    Real(1.0, 1.8, name='guide_w2'),
]


@use_named_args(search_space)
def objective(guide_w1, guide_w2):
    foldername = (
        f"{method}_{property_name}/"
        f"bayes_opt_w1_{guide_w1:.2f}_w2_{guide_w2:.2f}"
    )
    sdf_file = os.path.join(
        foldername, f"{method}_{property_name}_{n_mols}_{task}_step40000.sdf"
    )

    if not os.path.exists(sdf_file):
        os.makedirs(foldername, exist_ok=True)
        molecules = sample_molecules_with_auto_guidance(
            main_model,
            weak_model,
            n_mols=n_mols,
            multiple_values_to_one_property=multiple_values_to_one_property,
            number_of_atoms=no_atoms,
            property_name=property_name,
            properties_handle_method=prob_emb,
            n_timesteps=n_timesteps,
            normalization_file_path=normalization_file_path,
            guide_w1=guide_w1,
            guide_w2=guide_w2,
            dataset_name=args.dataset_name,
        )
        molecules_to_sdf(molecules, sdf_file)

    mae = compute_mae_from_sdf(
        sdf_file, property_name, multiple_values_to_one_property, dataset_name
    )
    print(
        f"w1={guide_w1:.2f}, w2={guide_w2:.2f},  MAE={mae:.4f}"
    )
    return mae


def main(args):
    # declare globals for use in objective()
    global main_model, weak_model
    global property_name, n_mols, task, prob_emb, normalization_file_path, n_timesteps
    global multiple_values_to_one_property, no_atoms, method

    # pull args into globals
    property_name           = args.property_name
    n_mols                  = args.n_mols
    task                    = args.task
    prob_emb                = args.prob_emb
    method                  = args.method
    normalization_file_path = args.normalization_file_path
    n_timesteps             = args.n_timesteps

    # load per-property data
    vals = np.load(f"{args.qm9_val_path}/train_half_sampled_values_{property_name}.npy")[:n_mols]
    multiple_values_to_one_property = [float(v) for v in vals]
    no_atoms = list(
        np.load(f"{args.qm9_val_path}/train_half_sampled_no_atoms_{property_name}.npy")[:n_mols]
    )

    # ckpt folders
    # main_folder = os.path.join(args.main_model_folder, property_name, prob_emb)
    # weak_folder = os.path.join(args.weak_model_folder, property_name, "runs_qm9_valid")

    # main_ckpt = get_model_ckpts(main_folder, args.main_ckpt_pattern)[0]
    # weak_ckpt = get_model_ckpts(weak_folder, args.weak_ckpt_pattern)[0]
    main_ckpt = args.main_model_folder
    weak_ckpt = args.weak_model_folder

    main_model = load_model_from_ckpt(main_ckpt)
    weak_model = load_model_from_ckpt(weak_ckpt)

    # Bayesian optimization
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func="EI",
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        random_state=args.random_seed,
    )

    # ─── Collect and dump results ────────────────────────────────────────────────
    results_dict = {
        'best_w1':  res.x[0],
        'best_w2':  res.x[1],
        'best_mae': float(res.fun),
        'guide_ws': np.array(res.x_iters),
        'maes':     res.func_vals,
    }
    os.makedirs("pkl-files", exist_ok=True)
    with open(f"pkl-files/{method}_results_{property_name}_step40000.pkl", "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--property_name", required=True)
    p.add_argument("--main_model_folder", required=True)
    p.add_argument("--weak_model_folder", required=True)
    p.add_argument("--prob_emb", default="concatenate_sum")
    p.add_argument("--main_ckpt_pattern", default="epoch*.ckpt")
    p.add_argument("--weak_ckpt_pattern", default="model-step=step=40000.ckpt")
    p.add_argument("--qm9_val_path", required=True)
    p.add_argument("--normalization_file_path", required=True)
    p.add_argument("--n_mols", type=int, default=1000)
    p.add_argument("--n_calls", type=int, default=30)
    p.add_argument("--n_initial", type=int, default=10)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--task", default="many2many")
    p.add_argument("--method", default="ag")
    p.add_argument("--n_timesteps", type=int, default=100)  # <-- added
    p.add_argument('--dataset-name', default='qme14s')
    args = p.parse_args()
    main(args)
