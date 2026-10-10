# Import core libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import re
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
output_dir = 'assets/exercise2'
os.makedirs(output_dir, exist_ok=True)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")

# 1. Load the dataset (from local directory)
dataset = load_dataset('Praxash1/AIgrow', split='train')
print(f"✓ Dataset loaded: {len(dataset)} entries found\n")


# 2. Parse the dataset
def parse_aigrow_entry(text):
    """Extract temperature and moisture from dataset text entries."""
    data = {}

    # Extract soil moisture
    moisture_match = re.search(r'Soil Moisture:\s*([\d.]+)', text)
    if moisture_match:
        data['moisture'] = float(moisture_match.group(1))

    # Extract temperature
    temp_match = re.search(r'Temperature:\s*([\d.]+)', text)
    if temp_match:
        data['temperature'] = float(temp_match.group(1))


    print(data)

    return data if len(data) == 2 else None


data_list = []
for item in dataset:
    parsed = parse_aigrow_entry(item['text'])
    if parsed:
        data_list.append(parsed)

print(f"✓ Parsed {len(data_list)} valid temperature-moisture readings")

# 3. Augment the dataset
# The original dataset is small (~16 samples), so we'll augment it by adding synthetic variations around the real measurements
print(f"✓ Augmenting dataset to create {400} samples for better visualization...")

original_data = pd.DataFrame(data_list)

# Generate augmented data by adding noise to original samples
augmented_data = []
target_size = 400

np.random.seed(42)
for _ in range(target_size):
    # Pick a random original sample
    base_idx = np.random.randint(0, len(original_data))
    base_temp = original_data.iloc[base_idx]['temperature']
    base_moisture = original_data.iloc[base_idx]['moisture']

    # Add realistic variation
    new_temp = base_temp + np.random.normal(0, 3)  # ±3°C variation
    new_moisture = base_moisture + np.random.normal(0, 5)  # ±5% variation

    # Clip to realistic ranges
    new_temp = np.clip(new_temp, 10, 40)
    new_moisture = np.clip(new_moisture, 5, 85)

    augmented_data.append({
        'temperature': new_temp,
        'moisture': new_moisture
    })

data_list = augmented_data

# 3. Convert to arrays
df = pd.DataFrame(data_list)
temperature_raw = df['temperature'].values
moisture_raw = df['moisture'].values

# 4. Normalize features to [0, 1] range for neural network
temperature_norm = (temperature_raw - 10) / 30  # 10-40°C -> 0-1
moisture_norm = moisture_raw / 90  # 0-90% -> 0-1

# 5. Combine features into matrix
X = np.column_stack([temperature_norm, moisture_norm])
X_tensor = torch.FloatTensor(X)

# 5. Create labels

# Create ELLIPTICAL labels (Goldilocks zone)
def create_elliptical_labels(temps, moistures, center=(0.5, 0.5), radii=(0.25, 0.20)):
    """
    Create elliptical pattern: healthy when BOTH temp and moisture are moderate
    """
    cx, cy = center
    rx, ry = radii
    distances = ((temps - cx) / rx) ** 2 + ((moistures - cy) / ry) ** 2
    return (distances <= 1).astype(int)

y_ellipse = create_elliptical_labels(temperature_norm, moisture_norm)
y_ellipse_tensor = torch.FloatTensor(y_ellipse)

# Create XOR labels (opposite corners pattern)
def create_xor_labels(temps, moistures, threshold=0.5):
    """
    Create XOR pattern: healthy when (cold AND wet) OR (hot AND dry)
    Unhealthy when (cold AND dry) OR (hot AND wet)
    """
    cold = temps < threshold
    wet = moistures < threshold
    # XOR logic: (cold AND wet) OR (hot AND dry)
    healthy = (cold & wet) | (~cold & ~wet)
    return healthy.astype(int)

y_xor = create_xor_labels(temperature_norm, moisture_norm)
y_xor_tensor = torch.FloatTensor(y_xor)


# TO CONCLUDE: Get a high-level dataset summary
print(f"\n{'='*60}")
print("Dataset Summary:")
print(f"{'='*60}")
print(f"Total samples: {len(X)}")
print(f"\nFeature ranges (raw):")
print(f"  Temperature: {temperature_raw.min():.1f}°C to {temperature_raw.max():.1f}°C")
print(f"  Moisture: {moisture_raw.min():.1f}% to {moisture_raw.max():.1f}%")
print(f"\nFeature ranges (normalized):")
print(f"  Temperature: {X[:, 0].min():.3f} to {X[:, 0].max():.3f}")
print(f"  Moisture: {X[:, 1].min():.3f} to {X[:, 1].max():.3f}")
print(f"\nClass distribution:")
print(f"✓ Ellipse pattern: {np.sum(y_ellipse == 1)} healthy, {np.sum(y_ellipse == 0)} unhealthy")
print(f"✓ XOR pattern: {np.sum(y_xor == 1)} healthy, {np.sum(y_xor == 0)} unhealthy\n")
print(f"{'='*60}")


# TODO: Implement three activation functions
# These helper functions will be used throughout the exercise

def apply_sigmoid(x):
    """
    Apply sigmoid activation: outputs range from 0 to 1.

    Hint: PyTorch has a built-in function for this common activation.
    Reference: https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    Args:
        x: Input tensor of any shape
    Returns:
        Tensor of same shape with sigmoid applied element-wise
    """
    return torch.sigmoid(x)  # Add your code here


def apply_tanh(x):
    """
    Apply tanh activation: outputs range from -1 to 1 (zero-centered).

    Hint: Tanh is the hyperbolic tangent function. PyTorch provides this.
    Reference: https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    Args:
        x: Input tensor of any shape
    Returns:
        Tensor of same shape with tanh applied element-wise
    """
    return torch.tanh(x)  # Add your code here


def apply_relu(x):
    """
    Apply ReLU activation: outputs negative values as 0, keeps positive values.

    Hint: ReLU stands for Rectified Linear Unit, and is calculated as max(0, x). PyTorch provides this.
    Reference: https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    Args:
        x: Input tensor of any shape
    Returns:
        Tensor of same shape with ReLU applied element-wise
    """
    return torch.relu(x)  # Add your code here


print("✓ Activation functions defined!")

# Create a range of input values
x = torch.linspace(-5, 5, 200)

# TODO: Compute activation function outputs
# HINT: Use the methods defined in the previous step, on the right output
sigmoid_output = apply_sigmoid(x)  # Add your code here
tanh_output = apply_tanh(x)  # Add your code here
relu_output = apply_relu(x)  # Add your code here

# Plot all three functions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Sigmoid
axes[0].plot(x.numpy(), sigmoid_output.numpy(), 'b-', linewidth=2)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Input', fontsize=11)
axes[0].set_ylabel('Output', fontsize=11)
axes[0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)\nOutput range: [0, 1]', fontsize=12, fontweight='bold')
axes[0].set_ylim(-0.2, 1.2)

# Tanh
axes[1].plot(x.numpy(), tanh_output.numpy(), 'g-', linewidth=2)
axes[1].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Input', fontsize=11)
axes[1].set_ylabel('Output', fontsize=11)
axes[1].set_title('Tanh: tanh(x)\nOutput range: [-1, 1]', fontsize=12, fontweight='bold')
axes[1].set_ylim(-1.2, 1.2)

# ReLU
axes[2].plot(x.numpy(), relu_output.numpy(), 'r-', linewidth=2)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('Input', fontsize=11)
axes[2].set_ylabel('Output', fontsize=11)
axes[2].set_title('ReLU: max(0, x)\nOutput range: [0, ∞)', fontsize=12, fontweight='bold')
axes[2].set_ylim(-0.5, 5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'activation_functions.png'), dpi=300, bbox_inuyhes='tight')
plt.show()

print("\nKey observations:")
print("• Sigmoid: Smooth S-curve, outputs between 0 and 1")
print("• Tanh: Smooth S-curve, outputs between -1 and 1 (zero-centered)")
print("• ReLU: Sharp cutoff at 0, unbounded positive outputs")


class SingleLayerNetwork(nn.Module):
    """
    A single-layer network that can create curved boundaries.
    It uses multiple parallel neurons and one activation function to create non-linear boundaries.

    Architecture: 2 inputs -> N neurons -> fixed aggregation -> 1 output

    The output aggregation uses fixed coefficients (all neurons contribute equally, i.e., AND logic),
    so only the first layer's weights need active configuration.
    """

    def __init__(self, n_neurons=4):
        # Single learnable layer: 2 inputs -> n_neurons
        self.n_neurons = n_neurons

        # Pre-configured weights for each activation function
        # Each activation needs different weights due to different output ranges
        self.weight_configs = {
            'sigmoid': {
                'W': [[20.0, -20.0, 0.0, 0.0],  # Input 1 (temp) weights for 4 neurons
                      [0.0, 0.0, 20.0, -20.0]],  # Input 2 (moisture) weights for 4 neurons
                'b': [-5.5, 15.0, -5.5, 15.0]  # Creates boundaries at 0.25 and 0.75
            },
            'tanh': {
                'W': [[10.0, -10.0, 0.0, 0.0],
                      [0.0, 0.0, 10.0, -10.0]],
                'b': [-2.5, 7.5, -2.5, 7.5]
            },
            'relu': {
                'W': [[8.0, -8.0, 0.0, 0.0],
                      [0.0, 0.0, 8.0, -8.0]],
                'b': [-2.0, 6.0, -2.0, 6.0]
            }
        }

    def forward(self, x, activation='sigmoid'):
        """
        Forward pass through the network.

        Args:
            X: Input tensor of shape (N, 2)
            activation: One of 'sigmoid', 'tanh', 'relu', or 'none'

        Returns:
            predictions: Final predictions (0 or 1)
            activations: Neuron activations (for visualization)
        """
        # Get the right weights for this activation
        config = self.weight_configs[activation]
        W = torch.FloatTensor(config['W'])
        b = torch.FloatTensor(config['b'])

        # Layer: Linear transformation
        z = torch.matmul(x, W) + b

        # Apply the selected activation function
        if activation == 'sigmoid':
            h = apply_sigmoid(z)
        elif activation == 'tanh':
            h = apply_tanh(z)
        elif activation == 'relu':
            h = apply_relu(z)
        else:  # 'none'
            h = z

        # Fixed aggregation: sum all neurons and apply threshold
        # This creates an "AND" logic: all neurons must agree for positive prediction
        # IMPORTANT: Clip ReLU outputs to [0,1] since it's unbounded, unlike sigmoid/tanh
        #            This can make the aggregation sum grow arbitrarily large, breaking the linear threshold logic
        if activation == 'relu':
            h = h.clip(0, 1)  # Clip ReLU's unbounded outputs for aggregation
        aggregation = h.sum(dim=1, keepdim=True) - (self.n_neurons - 0.5)

        # Convert to binary predictions
        # We compare the score against 0.0 because Sigmoid(0.0) = 0.5
        predictions = (aggregation > 0.0).float()

        return predictions, h

    def accuracy(self, X, y):
        """Calculate accuracy on given data."""
        predictions, _ = self.forward(X)
        return (predictions.squeeze() == y).float().mean().item()


# Create network
network = SingleLayerNetwork(n_neurons=4)

print("✓ Network created with pre-calculated weights for elliptical boundary")


# Visualize decision boundaries for different activation functions

def visualize_activation_comparison(network, X, y, activations=['sigmoid', 'tanh', 'relu']):
    """
    Compare decision boundaries for different activation functions.
    """
    # Create grid for decision boundary
    h = 0.01
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, activation in enumerate(activations):
        ax = axes[idx]

        # Get predictions for the grid using this activation
        # The forward method returns (predictions, activations)
        with torch.no_grad():
            preds_grid, _ = network.forward(grid_tensor, activation=activation)
            Z = preds_grid.numpy().reshape(xx.shape)

        # Plot decision regions
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                    colors=['#ffcccc', '#ccffcc'], alpha=0.4)
        ax.contour(xx, yy, Z, levels=[0.5],
                   colors='red', linewidths=2.5)

        # Plot data points
        for i, label in enumerate(['Unhealthy', 'Healthy']):
            mask = y == i
            color = 'red' if i == 0 else 'green'
            ax.scatter(X_tensor[mask, 0], X_tensor[mask, 1], c=color, label=label,
                       edgecolor='k', s=30, alpha=0.6, linewidth=0.5)

        # Calculate accuracy for this activation
        # Get predictions on the actual data points and compare with true labels
        with torch.no_grad():
            preds, _ = network.forward(X_tensor, activation=activation)
            accuracy = (preds.squeeze() == y_ellipse_tensor).float().mean().item()

        ax.set_title(f"{activation.upper()} Activation\nAccuracy: {accuracy:.1%}",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Temperature', fontsize=11)
        ax.set_ylabel('Moisture', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_comparison_ellipse.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


# Run the visualization
visualize_activation_comparison(network, X_tensor, y_ellipse_tensor)