import torch
import torch.nn as nn

# --- Model 1: Iterative Refinement ("step_out" / "iterative") ---
class ReasoningBlock(nn.Module):
    """
    A single reasoning step: Output -> Hidden Layer -> Output
    Uses a residual connection: New_Output = ReLU(Old_Output + Update)
    """
    def __init__(self, output_size, hidden_size):
        super(ReasoningBlock, self).__init__()
        self.output_to_hidden = nn.Linear(output_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, output_state): 
        residual = output_state
        
        hidden = self.relu(self.output_to_hidden(output_state))
        update = self.hidden_to_output(hidden) 
        
        new_output_state = self.relu(residual + update)
        return new_output_state

class IterativeRefinementNet(nn.Module):
    """
    Refines the output vector iteratively.
    Input -> Output_Guess -> Refine(Output) -> Refine(Output) -> ...
    """
    def __init__(self, input_size, hidden_sizes, num_iterations, output_size):
        super(IterativeRefinementNet, self).__init__()
        
        self.input_to_output = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        
        assert num_iterations == len(hidden_sizes), "Must provide a hidden_size for each iteration."
        
        self.reasoning_blocks = nn.ModuleList()
        for i in range(num_iterations):
            self.reasoning_blocks.append(
                ReasoningBlock(output_size, hidden_sizes[i])
            )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 1. Initial guess
        state = self.relu(self.input_to_output(x))
        
        # 2. Iterative refinement loop
        for block in self.reasoning_blocks:
            state = block(state)
            
        # 3. Final Output
        output = state
        return output