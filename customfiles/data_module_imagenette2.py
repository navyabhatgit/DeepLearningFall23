from fastdownload import FastDownload
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from pathlib import Path
import pytorch_lightning as pl
import torch
import os

class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_transform=None, test_transform=None,
                 batch_size=64, seed=42, fraction_test=0.5, small_subset = False, num_samples_small = 500):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        self.fraction_test = fraction_test
        self.num_samples_small = num_samples_small
        self.small_subset = small_subset
        self.n_workers = os.cpu_count() -1  # Number of workers for data loading

    def split_dataset(self, base_dataset):
        split_a_size = int(self.fraction_test * len(base_dataset))
        split_b_size = len(base_dataset) - split_a_size

        return torch.utils.data.random_split(
            base_dataset,
            [split_a_size, split_b_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def get_stratified_subset(self, dataset, labels, seed=None):
        _, subset_indices = train_test_split(
            range(len(labels)),  # Just indices, not the actual data
            test_size=self.num_samples_small,
            stratify=labels,
            random_state=seed
        )
        return Subset(dataset, subset_indices)


    def prepare_data(self):                
        d = FastDownload(base=self.data_dir, archive='archive', data='datasets')
        url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
        d.get(url)

    def setup(self, stage=None):
        self.trainset = datasets.ImageFolder(self.data_dir/'datasets/imagenette2/train', transform=self.train_transform)
        self.test_val_set = datasets.ImageFolder(self.data_dir/'datasets/imagenette2/val', transform=self.test_transform)

        # Directly accessing the targets attribute for labels
        test_val_labels = self.test_val_set.targets

        # Use Split for test and valid sets
        self.testset, self.validset = self.split_dataset(self.test_val_set)

        if self.small_subset:
            train_labels = self.trainset.targets
            self.trainset = self.get_stratified_subset(self.trainset, train_labels, self.seed)

            # Get labels for test and valid subsets
            test_labels = [test_val_labels[i] for i in self.testset.indices]
            self.testset = self.get_stratified_subset(self.testset, test_labels, self.seed)

            valid_labels = [test_val_labels[i] for i in self.validset.indices]
            self.validset = self.get_stratified_subset(self.validset, valid_labels, self.seed)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, 
                          drop_last=True, num_workers=self.n_workers )

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False,num_workers=self.n_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)