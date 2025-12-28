import torch
import torch.nn as nn

# Activation function mapping
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'mish': nn.Mish
}

class StandardFeedForwardNet(nn.Module):
    """
    A Standard Feed-Forward Network (MLP).
    Supports variable depth, width, activation functions, and dropout.
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.0):
        super(StandardFeedForwardNet, self).__init__()
        
        # Resolve activation function
        activation_fn_class = ACTIVATION_MAP.get(activation.lower())
        if activation_fn_class is None:
            raise ValueError(f"Unknown activation function: {activation}")
        
        layers = []
        
        if not hidden_sizes:
            # If no hidden sizes, it's just a simple linear model
            layers.append(nn.Linear(input_size, output_size))
        else:
            # 1. Input layer to first hidden layer
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(activation_fn_class()) 
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            # 2. Intermediate hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(activation_fn_class()) 
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            
            # 3. Last hidden layer to output layer
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)