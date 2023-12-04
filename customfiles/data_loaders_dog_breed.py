import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, Subset
from fastai.vision.all import PILImage
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class MyDataset(Dataset):
    """
    Custom dataset for handling image data along with optional labels.

    Attributes:
    img_folder (Path): Path to the directory containing the images.
    transform (callable, optional): A function/transform to apply to the images.
    has_labels (bool): Flag to check if the dataset has labels.
    img_names (list): List of image file names.
    labels (Series): Pandas Series containing labels, if available.
    label_to_idx (dict): Mapping from label names to numerical indices, if labels are available.
    """
    def __init__(self, img_folder, csv_file=None, transform=None, has_labels=True):
        self.img_folder = Path(img_folder)  # Convert to Path object for filesystem safety
        self.transform = transform  # Store the transform function
        self.has_labels = has_labels  # Flag to indicate presence of labels

        # If dataset has labels, read them from csv file
        if self.has_labels:
            df = pd.read_csv(csv_file)
            self.img_names = df["id"]
            self.labels = df["breed"]
            # Create a mapping from unique labels to indices
            unique_labels = self.labels.unique()
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            # Get image names from the folder if no labels
            self.img_names = [f.stem for f in self.img_folder.glob('*.jpg')]

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.img_names)

    def __getitem__(self, index):
        """Retrieve and optionally transform the item (image, label) at the given index."""
        # Construct the full image file path
        img_name_with_extension = f"{self.img_names[index]}.jpg"
        img_path = self.img_folder / img_name_with_extension

        # Load the image
        img = PILImage.create(img_path)

        # Apply the transformation, if any
        if self.transform:
            img = self.transform(img)

        # Return image and label if labels exist
        if self.has_labels:
            label_str = self.labels[index]
            label = self.label_to_idx[label_str]
            return img, label
        else:
            # If no labels, just return the image
            return img
        
        
def split_dataset(base_dataset, fraction, seed):
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size

    return torch.utils.data.random_split(
        base_dataset,
        [split_a_size, split_b_size],
        generator=torch.Generator().manual_seed(seed)
    )

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
    

def get_stratified_subset(dataset, labels, num_samples, seed=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=num_samples, random_state=seed)
    for _, subset_indices in sss.split(np.zeros(len(labels)), labels):
        break
    return Subset(dataset, subset_indices)


                                                                                                                
def get_loaders(img_folder, csv_file=None, train_transform=None, test_transform=None,  
                   fraction_train=0.8, seed=42, batch_size = 256, small_subset = False, num_samples_small=1000
                   ):
    train_val_set = MyDataset(img_folder=img_folder/'train', csv_file=csv_file)
    testset = MyDataset(img_folder=img_folder/'test', transform=test_transform, has_labels = False)
    trainset, validset = split_dataset(train_val_set, fraction = fraction_train, seed = seed)  
    trainset_transformed = TransformedSubset(trainset, train_transform)
    validset_transformed = TransformedSubset(validset, test_transform) 
    if  small_subset:
        train_val_labels = train_val_set.labels
        valid_labels = [train_val_labels[i] for i in validset.indices]
        train_labels = [train_val_labels[i] for i in trainset.indices]
        

        trainset_transformed = get_stratified_subset(trainset_transformed, train_labels, num_samples_small, seed=seed)
        validset_transformed = get_stratified_subset(validset_transformed, valid_labels, num_samples_small, seed=seed)
        
    train_loader = torch.utils.data.DataLoader(trainset_transformed, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset_transformed, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader




