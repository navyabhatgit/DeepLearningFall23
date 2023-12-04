import torch
import torch.nn as nn
class PytorchMLPSkip(nn.Module):
    """
    Custom Multilayer Perceptron (MLP) with skip connections class that inherits from PyTorch's nn.Module.

    Parameters:
    - num_features (int): The number of input features.
    - hidden_dim1 (int): The size of the first hidden layer.
    - hidden_dim2 (int): The size of the second hidden layer.
    - num_classes (int): The number of output classes.
    """
    def __init__(self, num_features, hidden_dim1, hidden_dim2, hidden_dim3, num_classes):
        super().__init__()

        # Flatten layer to convert input tensor shape from [batch_size, C, H, W] to [batch_size, C*H*W]
        self.flatten = nn.Flatten()

        # First hidden layer and its activation function
        self.hidden_layer1 = nn.Linear(num_features, hidden_dim1)
        self.layer1_activation = nn.ReLU()

        # Second hidden layer and its activation function
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer2_activation = nn.ReLU()

        # Second hidden layer and its activation function
        self.hidden_layer3 = nn.Linear(hidden_dim1 + hidden_dim2, hidden_dim3)
        self.layer3_activation = nn.ReLU()

        # Output layer that combines output from both hidden layers
        self.output_layer = nn.Linear(hidden_dim2 + hidden_dim3, num_classes)

    def forward(self, x):
        """
        Forward pass of the network with skip connection.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.
        """
        # Flatten the input
        x = self.flatten(x)

        # First hidden layer and its activation
        out1 = self.layer1_activation(self.hidden_layer1(x))

        # Second hidden layer and its activation
        out2 = self.layer2_activation(self.hidden_layer2(out1))

        # Concatenate output from both hidden layers
        concat_1_2 = torch.cat((out1, out2), dim=1)

        # Third hidden layer and its activation
        out3 = self.layer3_activation(self.hidden_layer3(concat_1_2))

        # Concatenate output from both hidden layers
        concat_2_3 = torch.cat((out2, out3), dim=1)

        # Final output layer
        return self.output_layer(concat_2_3)
