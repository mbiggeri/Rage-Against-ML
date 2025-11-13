# Iterative Refinement Network (IRN)

## Core Idea

This architecture is designed to simulate multi-step "reasoning" on a task. Instead of processing the input through one deep stack of layers, the model:

Makes an initial guess (an initial state) from the input.

Iteratively refines this state by passing it through a series of "reasoning blocks."

Makes its final prediction based on the fully refined state.

This is analogous to a human solving a problem: first, you get an initial idea, then you think about it, adjust your idea, think again, and then provide the final answer.

## Architecture

The network consists of three main parts:

A. Initial State Layer (The "Guess")

A single linear layer that takes the raw input (e.g., a flattened 784-pixel image) and maps it to the network's internal "state" space.

Path: Input $\rightarrow$ State

Formula: $\mathbf{h}^{(0)} = \text{ReLU}(\mathbf{W}_0 \mathbf{x} + \mathbf{b}_0)$

B. Reasoning Blocks (The "Refinement")

This is a series of $T$ identical modules that iteratively update the state. Each block has its own unique set of weights (it is not a standard recurrent layer with shared weights).

The key innovation is the residual connection, which is vital for stable training.

Path (per block): State $\rightarrow$ Hidden $\rightarrow$ State

Process:

Original State: $\mathbf{h}^{(t-1)}$

Update Path: The state is passed through two layers to create an "update" vector:
$\mathbf{u}_t = \text{HiddenToState}(\text{ReLU}(\text{StateToHidden}(\mathbf{h}^{(t-1)})))$

Refinement: The update is added to the original state (this is the residual connection):
$\mathbf{h}^{(t)} = \text{ReLU}(\mathbf{h}^{(t-1)} + \mathbf{u}_t)$

C. Final Output Layer (The "Answer")

After $T$ refinement steps, the final state $\mathbf{h}^{(T)}$ is passed through one last linear layer to be mapped to the final class outputs (e.g., 10 logits for MNIST).

Path: Final State $\rightarrow$ Output

Formula: $\mathbf{\hat{y}} = \mathbf{W}_{\text{out}} \mathbf{h}^{(T)} + \mathbf{b}_{\text{out}}$