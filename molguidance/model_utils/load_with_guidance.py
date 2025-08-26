from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
from molguidance.models.model_guidance_scale import ModelGuidance
from pathlib import Path


def model_from_config_with_guidance(config: dict, seed_ckpt: Path = None):
    """Create a model from configuration with guidance options.
    
    Args:
        config: The configuration dictionary
        seed_ckpt: Optional path to a checkpoint file to load from
        
    Returns:
        A configured model instance
    """
    # QM9 dataset Gaussian expansion parameters
    Gaussian_expansion_start_dict = {
        'mu': 0.0, 'alpha': 6.31, 'homo': -0.4286, 'lumo': -0.175, 
        'gap': 0.0246, 'cv': 6.002,
    }
    Gaussian_expansion_stop_dict = {
        'mu': 29.5564, 'alpha': 196.62, 'homo': -0.1017, 'lumo': 0.1935, 
        'gap': 0.6221, 'cv': 46.969,
    }

    # Extract configuration parameters
    atom_type_map = config['dataset']['atom_map']
    dataset_name = config['dataset'].get("dataset_name", 'qm9')
    conditional_generation = config['dataset']['conditioning']['enabled']
    property_embedding_dim = config['model_setting']['property_embedding_dim']
    gaussian_expansion = config['model_setting']['gaussian_expansion']['enabled']
    n_gaussians = config['model_setting']['gaussian_expansion']['n_gaussians']
    properties_handle_method = config['model_setting']['properties_handle_method']
    
    # Get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']

    # Set Gaussian expansion parameters if needed
    gaussian_start, gaussian_stop = None, None
    if gaussian_expansion and conditional_generation:
        properties = config['dataset']['conditioning']['property']
        gaussian_start = Gaussian_expansion_start_dict.get(properties)
        gaussian_stop = Gaussian_expansion_stop_dict.get(properties)

    # Get file paths
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_n_atoms_histogram.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    # Get guidance information
    guidance_config = config['guidance']
    with_guidance = guidance_config['enabled']
    guidance_type = guidance_config['guidance_type']

    # Select and instantiate the appropriate model
    model = None
    if guidance_type == 'classifier_free_guidance' and with_guidance:
        model_class = ClassifierFreeGuidance
        p_uncond = guidance_config['p_uncond']
        model = _create_model_instance(
            model_class, 
            guidance_type,
            seed_ckpt,
            atom_type_map, 
            n_atoms_hist_filepath,
            marginal_dists_file, 
            sample_interval, 
            mols_to_sample,
            config,
            property_embedding_dim,
            gaussian_expansion,
            gaussian_start,
            gaussian_stop,
            n_gaussians,
            conditional_generation,
            properties_handle_method,
            dataset_name,
            p_uncond=p_uncond
        )
    elif guidance_type == 'auto_guidance' and with_guidance:
        pass # no need to train for auto_guidance since we only do it on sampling
    elif guidance_type == 'model_guidance' and with_guidance:
        model_class = ModelGuidance
        p_uncond = guidance_config['p_uncond']
        min_scale = guidance_config['min_scale']
        max_scale = guidance_config['max_scale']
        mg_high = guidance_config['mg_high']
        data_ratio = guidance_config['data_ratio']
        mg_start_step = guidance_config['mg_start_step']
        use_scale_aware = guidance_config['use_scale_aware']
        model = _create_model_instance(
            model_class, 
            guidance_type,
            seed_ckpt,
            atom_type_map, 
            n_atoms_hist_filepath,
            marginal_dists_file, 
            sample_interval, 
            mols_to_sample,
            config,
            property_embedding_dim,
            gaussian_expansion,
            gaussian_start,
            gaussian_stop,
            n_gaussians,
            conditional_generation,
            properties_handle_method,
            dataset_name,
            drop_ratio=p_uncond,
            min_scale=min_scale,
            max_scale=max_scale,
            mg_high=mg_high,
            data_ratio=data_ratio,
            mg_start_step=mg_start_step,
            use_scale_aware=use_scale_aware
        )
    
    return model


def _create_model_instance(model_class, guidance_type, seed_ckpt, atom_type_map, n_atoms_hist_filepath, 
                         marginal_dists_file, sample_interval, mols_to_sample, config, 
                         property_embedding_dim, gaussian_expansion, gaussian_start, 
                         gaussian_stop, n_gaussians, conditional_generation, 
                         properties_handle_method, dataset_name, **kwargs):
    """Create a model instance with the given parameters.
    
    Args:
        model_class: The model class to instantiate
        seed_ckpt: Optional checkpoint to load from
        Other parameters: Configuration options for the model
        
    Returns:
        A configured model instance
    """
    common_kwargs = {
        'atom_type_map': atom_type_map,
        'n_atoms_hist_file': n_atoms_hist_filepath,
        'marginal_dists_file': marginal_dists_file,
        'sample_interval': sample_interval,
        'n_mols_to_sample': mols_to_sample,
        'vector_field_config': config['vector_field'],
        'interpolant_scheduler_config': config['interpolant_scheduler'],
        'lr_scheduler_config': config['lr_scheduler'],
        'property_embedding_dim': property_embedding_dim,
        'gaussian_expansion': gaussian_expansion,
        'gaussian_start': gaussian_start,
        'gaussian_stop': gaussian_stop,
        'n_gaussians': n_gaussians,
        'conditional_generation': conditional_generation,
        'properties_handle_method': properties_handle_method,
        'dataset_name': dataset_name,
        **config['mol_fm']
    }
    
    if guidance_type == 'classifier_free_guidance':
        common_kwargs.update({
            'p_uncond': kwargs['p_uncond']
        })
    elif guidance_type == 'model_guidance':
        common_kwargs.update({
            'drop_ratio': kwargs['drop_ratio'],
            'min_scale': kwargs['min_scale'],
            'max_scale': kwargs['max_scale'],
            'mg_high': kwargs['mg_high'],
            'data_ratio': kwargs['data_ratio'],
            'mg_start_step': kwargs['mg_start_step'],
            'use_scale_aware': kwargs['use_scale_aware'],
        })

    if seed_ckpt is not None:
        return model_class.load_from_checkpoint(seed_ckpt, **common_kwargs)
    else:
        return model_class(**common_kwargs)