import torch
import torch.nn as nn

# Activation function mapping ---
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU
}

class ReasoningBlock(nn.Module):
    """
    A single reasoning step: Output -> Hidden Layer -> Output
    Uses a residual connection: New_Output = Activation(Old_Output + Update)
    """
    def __init__(self, output_size, hidden_size, activation_str='relu'):
        super(ReasoningBlock, self).__init__()
        self.output_to_hidden = nn.Linear(output_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        activation_fn = ACTIVATION_MAP.get(activation_str.lower())
        if activation_fn is None:
            raise ValueError(f"Unknown activation function: {activation_str}")
        
        # Use the selected activation for both hidden and residual paths
        self.activation = activation_fn()

    def forward(self, output_state): 
        residual = output_state
        
        # Activation is used on the hidden layer
        hidden = self.activation(self.output_to_hidden(output_state))
        update = self.hidden_to_output(hidden) 
        
        # Activation is also used on the residual connection
        new_output_state = self.activation(residual + update)
        return new_output_state

class IterativeRefinementNet(nn.Module):
    """
    Refines the output vector iteratively.
    Input -> Output_Guess -> Refine(Output) -> Refine(Output) -> ...
    """
    def __init__(self, input_size, hidden_sizes, num_iterations, output_size, activation_str='relu'):
        super(IterativeRefinementNet, self).__init__()
        
        self.input_to_output = nn.Linear(input_size, output_size)
        
        activation_fn = ACTIVATION_MAP.get(activation_str.lower())
        if activation_fn is None:
            raise ValueError(f"Unknown activation function: {activation_str}")
            
        # Use selected activation for the initial guess
        self.activation = activation_fn()
        
        assert num_iterations == len(hidden_sizes), "Must provide a hidden_size for each iteration."
        
        self.reasoning_blocks = nn.ModuleList()
        for i in range(num_iterations):
            self.reasoning_blocks.append(
                # Pass the activation_str down to each reasoning block
                ReasoningBlock(output_size, hidden_sizes[i], activation_str=activation_str)
            )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 1. Initial guess (using selected activation)
        state = self.activation(self.input_to_output(x))
        
        # 2. Iterative refinement loop
        for block in self.reasoning_blocks:
            state = block(state)
            
        # 3. Final Output
        # Note: The final output is the result of the last block's
        # activation function (e.g., ReLU), which is not ideal for
        # regression tasks that can have negative outputs.
        output = state
        return output