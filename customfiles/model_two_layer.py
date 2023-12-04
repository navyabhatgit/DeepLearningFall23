import torch.nn as nn

class TwoLayerMLP(nn.Module):
    """
    Custom Multilayer Perceptron (MLP) class that inherits from PyTorch's nn.Module.

    Parameters:
    - num_features (int): The number of input features.
    - hidden_dim1 (int): The size of the first hidden layer.
    - hidden_dim2 (int): The size of the second hidden layer.
    - num_classes (int): The number of output classes.
    """
    def __init__(self, num_features, hidden_dim1, hidden_dim2, num_classes):
        super().__init__()

        # Define the sequence of layers for the MLP
        self.layers = nn.Sequential(
            # Flatten the input tensor from [batch_size, Channels, Height, Width] to [batch_size, Channels*Height*Width]
            nn.Flatten(start_dim=1, end_dim=-1),

            # First fully connected layer, followed by ReLU activation
            nn.Linear(num_features, hidden_dim1),
            nn.ReLU(),

            # Second fully connected layer, followed by ReLU activation
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),

            # Output layer
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.
        """
        out = self.layers(x)
        return out