import pandas as pd
from pathlib import Path
from torchvision import datasets
from torch.utils.data import Dataset
import torch
import random
from collections import defaultdict
import os

class TransformedSubset(Dataset):
    """
    A Dataset wrapper that applies a transform to a subset of a dataset.

    Attributes:
    subset (Dataset): The subset of data to which the transform will be applied.
    transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset  # The original data subset
        self.transform = transform  # The transform function to apply on the data

    def __getitem__(self, index):
        """
        Retrieve and optionally transform the item (image, label) at the given index.

        Parameters:
        index (int): Index of the item to retrieve.

        Returns:
        tuple: Transformed image and label pair.
        """
        # Retrieve original data
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
        
def split_dataset(base_dataset, fraction, seed):
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size

    return torch.utils.data.random_split(
        base_dataset,
        [split_a_size, split_b_size],
        generator=torch.Generator().manual_seed(seed)
    )

def get_stratified_subset(dataset, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
        
    # Step 1: Identify label distribution
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)
    
    # Step 2: Calculate proportions and initialize subset indices list
    proportions = {label: len(indices) / len(dataset) for label, indices in label_to_indices.items()}
    subset_indices = []
    
    # Step 3: Sample according to proportion
    for label, indices in label_to_indices.items():
        num_samples_for_label = round(proportions[label] * num_samples)
        subset_indices += random.sample(indices, num_samples_for_label)
    
    # Step 4: Combine samples
    return torch.utils.data.Subset(dataset, subset_indices)
                                                                                                               
                                                                                                                
def get_loaders(data_folder, train_transform=None, test_transform=None,  
                   fraction_train=0.8, seed=42, batch_size = 256, small_subset = False, num_samples_small=1000
                   ):
    train_val_set = datasets.FashionMNIST(root=data_folder, train=True, download=True)
    testset = datasets.FashionMNIST(root=data_folder, train=False, download=True, transform=test_transform)

    trainset, validset = split_dataset(train_val_set, fraction = fraction_train, seed = seed)  

    trainset_transformed = TransformedSubset(trainset, train_transform)
    validset_transformed = TransformedSubset(validset, test_transform) 

    if  small_subset:
        trainset_transformed = get_stratified_subset(trainset_transformed, num_samples=num_samples_small, seed=seed)
        validset_transformed = get_stratified_subset(validset_transformed, num_samples=num_samples_small, seed=seed)
        
        
    train_loader = torch.utils.data.DataLoader(trainset_transformed, batch_size=batch_size, shuffle=True, 
                                               num_workers=os.cpu_count()-1)
    valid_loader = torch.utils.data.DataLoader(validset_transformed, batch_size=batch_size, shuffle=False, 
                                               num_workers=os.cpu_count()-1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                              num_workers=os.cpu_count()-1)
    return train_loader, valid_loader, test_loader



