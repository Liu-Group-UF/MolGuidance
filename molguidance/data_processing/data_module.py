import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import dgl

from molguidance.data_processing.dataset import collate
from molguidance.data_processing.dataset import MoleculeDataset
from molguidance.data_processing.samplers import SameSizeMoleculeSampler, SameSizeDistributedMoleculeSampler

class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, dm_prior_config: dict, batch_size: int, num_workers: int = 0, distributed: bool = False, max_num_edges: int = 40000):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = dm_prior_config
        self.max_num_edges = max_num_edges
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            # self.train_dataset = self.load_dataset('train')
            self.val_dataset = self.load_dataset('val')
            self.test_dataset = self.load_dataset('test')
            if self.dataset_config['use_first_half_training_set'] == True:
                self.train_dataset = self.load_dataset('train_a')
            else:
                self.train_dataset = self.load_dataset('train_b')

    def load_dataset(self, dataset_name: str):
        return MoleculeDataset(dataset_name, self.dataset_config, prior_config=self.prior_config)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)

        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, 
                                batch_size=self.batch_size*2, 
                                shuffle=False, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size*2, 
                                shuffle=False, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)
        return dataloader
