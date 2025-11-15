import torch
import torch.nn as nn

# Activation function mapping ---
# Defined here so both classes can use it
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU
}

class StandardFeedForwardNet(nn.Module):
    """
    A standard "deep" feed-forward network (MLP).
    The architecture is built from the hidden_sizes list.
    Accepts an 'activation_str' to define the hidden activation function.
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation_str='relu'):
        super(StandardFeedForwardNet, self).__init__()
        
        # --- NEW: Activation function mapping ---
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        activation_fn = activations.get(activation_str.lower())
        if activation_fn is None:
            raise ValueError(f"Unknown activation function: {activation_str}")
        
        layers = []
        
        if not hidden_sizes:
            # If no hidden sizes, it's just a simple linear model
            layers.append(nn.Linear(input_size, output_size))
        else:
            # 1. Input layer to first hidden layer
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(activation_fn()) # Use selected activation
            
            # 2. Intermediate hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(activation_fn()) # Use selected activation
            
            # 3. Last hidden layer to output layer
            # (No activation here, as required for both regression (MSE)
            # and classification (CrossEntropyLoss))
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if it's, e.g., an image
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)