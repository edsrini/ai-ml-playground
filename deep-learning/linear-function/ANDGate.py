# Import core libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
output_dir = 'assets/_SOLUTION-exercise1'
os.makedirs(output_dir, exist_ok=True)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")

# Truth table inputs shared by both gates
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Corresponding outputs for each logic gate
y_and = np.array([0, 0, 0, 1])
y_xor = np.array([0, 1, 1, 0])

# Create side-by-side plots for AND and XOR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color and label settings for class 0 and class 1
colors = ['#ff7f0e', '#1f77b4']
labels_text = ['Output = 0', 'Output = 1']


def plot_gate(ax, X, y, title, subtitle):
    """
    Plot a truth table for a binary gate:
    - scatters points for each input pair
    - colors them according to their output class
    - annotates each point with (x1,x2) and output
    """

    # Plot points for each output class (0 and 1)
    for cls in (0, 1):
        mask = (y == cls)
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[cls],
            s=300,
            label=labels_text[cls],
            edgecolors='black',
            linewidth=2
        )

    # Annotate each point with its input and output value
    for (x1, x2), out in zip(X, y):
        ax.text(
            x1, x2 - 0.18,
            f"({x1},{x2})\n→{out}",
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    # Apply consistent visual formatting to the axes
    ax.set_xlabel('Input 1', fontsize=12)
    ax.set_ylabel('Input 2', fontsize=12)
    ax.set_title(f'{title}\n({subtitle})', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])


# Plot AND and XOR truth tables using the shared function
plot_gate(axes[0], X, y_and, 'AND Gate', 'True if BOTH inputs are 1')
plot_gate(axes[1], X, y_xor, 'XOR Gate', 'True if inputs DIFFER')

# Layout and export
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'truth_tables.png'), dpi=300, bbox_inches='tight')
plt.show()

# Summary text
print("\nKey observation:")
print("• AND gate: Classes are LINEARLY SEPARABLE")
print("• XOR gate: Classes are NOT linearly separable")


class Perceptron:
    """
    A simple perceptron for binary classification.
    Computes: z = w1*x1 + w2*x2 + b, then output = 1 if z > 0 else 0
    """

    def __init__(self, input_size=2):
        """
        Initialize perceptron with random weights and bias.

        Args:
            input_size: Number of input features (default 2 for logic gates)
        """
        # TODO: Initialize weights as a tensor of shape (input_size,)
        # HINT: You need a 1D tensor with input_size random values.
        # Reference: https://docs.pytorch.org/docs/stable/torch.html#random-sampling
        self.weights = torch.randn(input_size)  # Add your code here

        # TODO: Initialize bias as a single value tensor
        # HINT: The bias is just one (random) number, but it needs to be a tensor. What shape creates a tensor with exactly one element?
        # Reference: https://docs.pytorch.org/docs/stable/torch.html#random-sampling
        self.bias = torch.randn(1)  # Add your code here

    def set_weights(self, weights, bias):
        """
        Manually set the weights and bias.
        Useful for testing specific configurations.
        """
        # TODO: Convert inputs to torch tensors and store them
        # HINT: You're given weights as a list (e.g., [1.0, 1.0]). Which PyTorch function converts a Python list into a float tensor?
        # Reference: https://docs.pytorch.org/docs/stable/tensors.html
        self.weights = torch.FloatTensor(weights)  # Add your code here
        self.bias = torch.FloatTensor([bias])  # Add your code here

    def forward(self, X):
        """
        Compute the perceptron output for input X.

        Args:
            X: Input tensor of shape (N, input_size)

        Returns:
            predictions: Binary predictions (0 or 1) of shape (N,)
            z: Raw activations before step function of shape (N,)
        """
        # TODO: Compute the weighted sum z = X @ weights + bias
        # HINT: Think about the perceptron equation: z = w₁×x₁ + w₂×x₂ + b. In matrix form, this is a dot product. What do you multiply, and what do you add?
        # Reference: https://docs.pytorch.org/docs/stable/tensors.html
        z = torch.matmul(X, self.weights) + self.bias  # Add your code here

        # TODO: Apply step activation function
        # HINT: Output should be 1 if z > 0, else 0. How do you convert True/False to 1.0/0.0?
        # Reference: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.float.html
        predictions = (z > 0).float()  # Add your code here

        return predictions, z

    def accuracy(self, X, y):
        """
        Calculate accuracy on given data.

        Args:
            X: Input features
            y: True labels

        Returns:
            accuracy: Fraction of correct predictions
        """
        # TODO: Get predictions using forward pass
        # HINT: You already wrote the forward method above, so you can just call it with your input.
        # Note that forward() returns two things - which one do you need for accuracy?
        predictions, _ = self.forward(X)  # Add your code here

        # TODO: Compare predictions with true labels and compute mean
        # HINT: Chain three operations: (1) compare predictions to true labels with ==, (2) convert the boolean result to numbers, (3) take the average.
        # The mean of 0s and 1s gives you the fraction that are correct!
        # Reference: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html?utm_source=chatgpt.com#more-math-with-tensors
        accuracy = (predictions == y).float().mean()  # Add your code here

        return accuracy.item()


# Test the Perceptron class with random initialization
test_perceptron = Perceptron(input_size=2)
print("Perceptron class implemented!")
print(f"\nInitial random weights: {test_perceptron.weights}")
print(f"Initial random bias: {test_perceptron.bias}")

# Test on AND gate data
X_and_tensor = torch.FloatTensor(X)
y_and_tensor = torch.FloatTensor(y_and)
preds, z = test_perceptron.forward(X_and_tensor)
print(f"\nRandom predictions on AND gate: {preds}")
print(f"True labels: {y_and_tensor}")
print(f"Random accuracy: {test_perceptron.accuracy(X_and_tensor, y_and_tensor):.1%}")

# Create a perceptron for AND gate
and_perceptron = Perceptron(input_size=2)

# TODO: Find weights and bias that solve the AND gate
# HINT: **Step-by-step approach**:
#   > 1. Try equal weights first: w₁=1, w₂=1
#   > 2. Calculate z for each input:
#   >    - (0,0): z = 1×0 + 1×0 + b = b
#   >    - (0,1): z = 1×0 + 1×1 + b = 1 + b
#   >    - (1,0): z = 1×1 + 1×0 + b = 1 + b
#   >    - (1,1): z = 1×1 + 1×1 + b = 2 + b
#   > 3. You need: (0,0), (0,1), (1,0) → z < 0 and (1,1) → z > 0
#   > 4. What value of b satisfies: b < 0, (1+b) < 0, but (2+b) > 0?
and_weights = [1.0, 1.0]  # Add your values here
and_bias = -1.5  # Add your value here

and_perceptron.set_weights(and_weights, and_bias)

# Test your solution
predictions, z_values = and_perceptron.forward(X_and_tensor)
accuracy = and_perceptron.accuracy(X_and_tensor, y_and_tensor)

print("AND Gate Solution:")
print(f"Weights: {and_weights}")
print(f"Bias: {and_bias}")
print(f"\nTruth Table:")
print("Input   | Expected | Predicted | z value")
print("-" * 45)
for i in range(len(X)):
    print(f"{X[i]} | {y_and[i]:.0f}        | {predictions[i]:.0f}         | {z_values[i].item():.2f}")

print(f"\nAccuracy: {accuracy:.1%}")

if accuracy == 1.0:
    print("✓ Perfect! You've solved the AND gate - every point is on the correct side!")
else:
    print("✗ Not quite. Try adjusting the weights and bias.")
    print("Hint: The decision boundary should separate (1,1) from all other points.")