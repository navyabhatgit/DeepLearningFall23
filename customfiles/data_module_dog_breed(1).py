from fastdownload import FastDownload
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import pytorch_lightning as pl
import torch
import os
import pandas as pd
from fastai.vision.all import PILImage



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





class DogBreedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, kaggle_api_folder, train_transform=None, test_transform=None,
                 batch_size=64, seed=42, fraction_train=0.8, small_subset = False, num_samples_small = 500):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.kaggle_api_folder= kaggle_api_folder
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        self.fraction_train = fraction_train
        self.num_samples_small = num_samples_small
        self.small_subset = small_subset
        self.n_workers = os.cpu_count() -1  # Number of workers for data loading
       
    def split_dataset(self, base_dataset):
        split_a_size = int(self.fraction_train * len(base_dataset))
        split_b_size = len(base_dataset) - split_a_size

        return torch.utils.data.random_split(
            base_dataset,
            [split_a_size, split_b_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
    
    def get_stratified_subset(self, dataset, labels,):
        _, subset_indices = train_test_split(
            range(len(labels)),  # Just indices, not the actual data
            test_size=self.num_samples_small,
            stratify=labels,
            random_state=self.seed
            )
        return Subset(dataset, subset_indices)
    
    def prepare_data(self): 
        os.environ['KAGGLE_CONFIG_DIR'] = str(self.kaggle_api_folder)
        from kaggle.api import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('dog-breed-identification', 
                                       path=self.data_dir/'archive')
        
        # !kaggle competitions download -c dog-breed-identification -p {self.data_folder/'archive'}
        d = FastDownload(base=self.data_dir, archive='archive', data='datasets')
        url = self.data_dir/'archive/dog-breed-identification.zip'
        d.extract(url)
        

    def setup(self, stage=None):
        data_path = self.data_dir/'datasets/dog-breed-identification'
        self.train_val_set = MyDataset(img_folder=data_path/'train', csv_file=data_path/'labels.csv')

        self.testset = MyDataset(img_folder=data_path/'test', transform=self.test_transform, has_labels = False)
        
        self.trainset, self.validset = self.split_dataset(self.train_val_set)  
        self.trainset_transformed = TransformedSubset(self.trainset, self.train_transform)
        self.validset_transformed = TransformedSubset(self.validset, self.test_transform) 
        if  self.small_subset:
            train_val_labels = self.train_val_set.labels
            valid_labels = train_val_labels[self.validset.indices]
            train_labels = train_val_labels[self.trainset.indices]
            self.trainset_transformed = self.get_stratified_subset(self.trainset_transformed, train_labels)
            self.validset_transformed = self.get_stratified_subset(self.validset_transformed, valid_labels) 
                                                      

    def train_dataloader(self):
        return DataLoader(self.trainset_transformed, batch_size=self.batch_size, shuffle=True, 
                          drop_last=True, num_workers=self.n_workers )

    def val_dataloader(self):
        return DataLoader(self.validset_transformed, batch_size=self.batch_size, shuffle=False,num_workers=self.n_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
    
