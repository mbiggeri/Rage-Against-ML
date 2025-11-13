import torch
import torch.nn as nn

# --- Model 2: Standard Feed-Forward ("classic" / "standard") ---
class StandardFeedForwardNet(nn.Module):
    """
    A standard "deep" feed-forward network (MLP).
    The architecture is built from the hidden_sizes list.
    e.g., hidden_sizes=[256, 128] will create:
    Input -> Linear(In, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, Out)
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(StandardFeedForwardNet, self).__init__()
        
        layers = []
        
        if not hidden_sizes:
            # If no hidden sizes, it's just a simple linear model
            layers.append(nn.Linear(input_size, output_size))
        else:
            # 1. Input layer to first hidden layer
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            
            # 2. Intermediate hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
            
            # 3. Last hidden layer to output layer
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if it's, e.g., an image
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)