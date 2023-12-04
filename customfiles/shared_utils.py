
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import torch
import pandas as pd

def check_loader(loader):
    """
    Inspects the first few batches from a DataLoader object to report shapes and optionally labels.

    Parameters:
    loader (torch.utils.data.DataLoader): The DataLoader to inspect.
    """
    num_epochs = 1  # Only going through one epoch for this check
    has_labels = None  # Initialize variable to store whether the dataset has labels or not

    for epoch in range(num_epochs):  # Loop through epochs (just one in this case)

        # Loop through mini-batches
        for idx, batch in enumerate(loader):
            if idx >= 3:  # Stop after checking 3 batches
                break

            # Determine if the dataset has labels based on the length of the returned tuple
            if len(batch) == 2:
                has_labels = True  # The dataset has labels
                image, label = batch  # Unpack batch into images and labels
            else:
                has_labels = False  # The dataset has no labels
                image = batch[0]  # Only images are in the batch

            # Print shape information for the images and optionally labels
            if has_labels:
                print(f" Batch Number: {idx+1} | Batch size: {image.shape[0]} | x shape: {image.shape} | y shape: {label.shape}")
            else:
                print(f" Batch Number: {idx+1} | Batch size: {image.shape[0]} | x shape: {image.shape}")

    # After the loop, print labels if available
    if has_labels:
        print(f"\nLabels from current batch \n {label}")
    else:
        print("\nNo labels in this dataset.")

def check_transforms(loader):
    for input, target in loader:
        print(f'shape of inputs is :{input.shape}')
        print(f'\nmax input value  :{input.max()}')
        print(f'\nmin input value  :{input.min()}')
        print(f'\nmean input value  :{input.mean()}')
        print(f'\nstd input value  :{input.std()}')
        print(f'\nshape of targets is :{target.shape}')
        print(f'\ninputs  :{input[0, 0, 5:10, 5:10]}')

        break

def check_label_dist(loader):
    """
    Check and display the distribution of labels in a DataLoader.

    Args:
        loader (torch.utils.data.DataLoader): The DataLoader containing batches of data.

    Returns:
        None
    """
    # Initialize a Counter object to store the counts of each label.
    label_counter = Counter()

    # Loop through batches of data from the DataLoader.
    for images, labels in loader:
        # Update the label counter with the list of labels from the current batch.
        label_counter.update(labels.tolist())

    # Print the sorted distribution of labels.
    print("\n Label distribution:")
    return sorted(label_counter.items())

def show_confusion_matrix(labels, predictions, classes):

    # Compute the confusion matrix between actual labels and predicted labels
    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())

    # Create a ConfusionMatrixDisplay object for visualization with class labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # Create the confusion matrix plot
    ax = disp.plot().ax_

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Access the Matplotlib figure from the Axes object and set its size

    # Get the current figure
    fig = plt.gcf()

    # Remove the last (existing) color bar
    plt.delaxes(fig.axes[-1])
    ax.figure.set_size_inches(8, 8)

    # Modify the colorbar size
    ax.figure.colorbar(ax.images[0], ax=ax, shrink=0.65)  # 'shrink' parameter adjusts the size

    plt.show()

def compute_accuracy(model, loader, device=None):

    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    running_correct = 0.0
    total_examples = 0

    for inputs, labels in loader:

        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        predictions = torch.argmax(outputs, dim=1)

        correct = torch.sum(labels == predictions)
        running_correct += correct
        total_examples += len(labels)

    return running_correct / total_examples



def plot_losses_acc(file):
    df = pd.read_csv(file)
    
    # Plotting Accuracy Metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['step'][df['train_acc'].notna()], df['train_acc'].dropna(), label='Train Accuracy')
    plt.plot(df['step'][df['val_acc'].notna()], df['val_acc'].dropna(), label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plotting Loss Metrics
    ax1 = plt.subplot(1, 2, 2)
    plt.plot(df['step'][df['train_loss_step'].notna()], df['train_loss_step'].dropna(), label='Stepwise Train Loss')
    plt.plot(df['step'][df['val_loss'].notna()], df['val_loss'].dropna(), label='Validation Loss')
    plt.plot(df['step'][df['train_loss_epoch'].notna()], df['train_loss_epoch'].dropna(), label='Epoch-wise Train Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(train_losses, train_metrics=None, val_losses=None, val_metrics=None):
    """
    Plot training and validation loss and metrics over epochs.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_metrics (list): List of training metrics (e.g., accuracy) for each epoch.
        val_losses (list, optional): List of validation losses for each epoch.
        val_metrics (list, optional): List of validation metrics for each epoch.

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation losses
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    if val_losses:
        plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation metrics (if available)
    if train_metrics is not None:
        plt.figure()
        plt.plot(epochs, train_metrics, label="Train")
        if val_metrics:
            plt.plot(epochs, val_metrics, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()