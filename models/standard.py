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

class ReadoutAdapter(nn.Module):
    """
    Adapter layer to format the output of the base model for a specific task.
    For regression, it's typically Identity or a Linear projection if dims mismatch.
    """
    def __init__(self, input_size, output_size, task_type='regression'):
        super(ReadoutAdapter, self).__init__()
        self.task_type = task_type
        
        # If input != output, we might need a projection, 
        # BUT usually the Base Model's last layer already outputs 'output_size'.
        # Assuming Base Model outputs [Batch, Output_Size]
        
        # In this project, StandardFeedForwardNet(..., output_size=OUTPUT_SIZE) 
        # already produces the correct dimension.
        # So ReadoutAdapter might just be an Identity pass-through or final activation.
        
        self.output_layer = nn.Identity()
            
    def forward(self, x):
        return self.output_layer(x)

class ModelWithHead(nn.Module):
    """
    Wraps a base backbone (feature extractor) and a task-specific head (readout).
    """
    def __init__(self, base_model, head):
        super(ModelWithHead, self).__init__()
        self.base = base_model
        self.head = head
        
    def forward(self, x):
        features = self.base(x)
        return self.head(features)
