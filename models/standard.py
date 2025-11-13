import torch
import torch.nn as nn

# --- Model 2: Standard Feed-Forward ("classic" / "standard") ---
class StandardFeedForwardNet(nn.Module):
    """
    A standard "deep" feed-forward network (FFNN) for comparison.
    """
    def __init__(self, input_size, state_size, hidden_sizes, num_iterations, output_size):
        super(StandardFeedForwardNet, self).__init__()
        
        assert num_iterations == len(hidden_sizes), "Must provide a hidden_size for each iteration."
        
        layers = []
        
        # 1. Input layer
        layers.append(nn.Linear(input_size, state_size))
        layers.append(nn.ReLU())
        
        current_size = state_size
        
        # 2. Create the "deep" stack of layers
        for i in range(num_iterations):
            hidden_size = hidden_sizes[i]
            
            # Layer 1
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            
            # Layer 2
            layers.append(nn.Linear(hidden_size, state_size))
            layers.append(nn.ReLU())
            
            current_size = state_size
        
        # 3. Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)