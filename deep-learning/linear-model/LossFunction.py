# Import core libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")

# 1. Load the dataset
dataset = load_dataset('VerisimilitudeX/Weather', split='train')
print(f"✓ Dataset loaded: {len(dataset)} weather observations\n")

# Convert to pandas
df = pd.DataFrame(dataset)

# 2. Create binary rain labels (1 if precipitation > 0, else 0)
df['rain'] = (df['Data.Precipitation'] > 0).astype(int)

# 3. Extract features: temperature, wind speed
df_clean = df[['Data.Temperature.Avg Temp', 'Data.Wind.Speed', 'rain']].dropna()

print(f"✓ Created {len(df_clean)} samples with rain labels")
print(f"  Rain days: {df_clean['rain'].sum()} ({df_clean['rain'].mean()*100:.1f}%)")
print(f"  No-rain days: {(1-df_clean['rain']).sum()} ({(1-df_clean['rain'].mean())*100:.1f}%)")

# 4. Sample a batch of data for our experiments
sample_size = 100
sample_df = df_clean.sample(n=sample_size, random_state=42)

# 5. Convert features and labels to tensors
X = sample_df[['Data.Temperature.Avg Temp', 'Data.Wind.Speed']].values
y = sample_df['rain'].values

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 6. Normalize features (important for neural networks)
X_tensor = (X_tensor - X_tensor.mean(dim=0)) / X_tensor.std(dim=0)

print(f"✓ Prepared batch of {sample_size} samples")
print(f"  Features shape: {X_tensor.shape}")
print(f"  Labels shape: {y_tensor.shape}")

# Create a simple model that outputs raw logits (no sigmoid activation)
class RainPredictor(nn.Module):
    def __init__(self, input_size=2):
        super(RainPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Output layer: NO activation = raw logits
        )

    def forward(self, x):
        return self.network(x).squeeze()  # Remove extra dimension


# Create model and get predictions
model = RainPredictor()

# Get predictions (logits) from the untrained model
with torch.no_grad():
    logits = model(X_tensor)

print("✓ Model created and predictions generated\n")
print(f"Sample logits (first 10):")
print(logits[:10].numpy())
print(f"\nLogit range: [{logits.min():.2f}, {logits.max():.2f}]")

# TODO: Create a loss function for binary classification and compute the loss
# HINT: PyTorch has several loss functions for classification in the nn module.
# What seems appropriate for binary classification (two classes: rain vs no-rain)?
# Reference: https://pytorch.org/docs/stable/nn.html#loss-functions

# Create your loss function by calling the right pytorch class
loss_fn = nn.BCELoss()  # Add your code here

# Compute the loss with predicted logits and true y tensor labels
try:
    loss_value = loss_fn(logits, y_tensor)  # Add your code here
    print(f"   Loss function {loss_fn} successfully called.")
    print(f"\n→ Continue to Step 4.2 to verify this is the right approach and see alternatives!")

except Exception as e:
    print(f"⚠︎  Error occurred: {e}")
    print(f"\n→ Continue to Step 4.2 for guided troubleshooting based on your error.")

# Path A: Fix the input mismatch
# TODO: Convert logits to probabilities, then compute BCELoss
# HINT: You need an activation function that maps any real number to the range [0, 1].
# Reference: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

# Step 1: Convert logits to probabilities using the right activation function
probabilities = torch.sigmoid(logits) # Add your code here

# Step 2: Re-create your loss (expects probabilities as input)
probability_based_loss = nn.BCELoss() # Add your code here

# Step 3: Compute loss with the probabilities and true y tensor labels
try:
    loss_manual_fix = probability_based_loss(probabilities, y_tensor) # Add your code here

    print(f"✓ Fixed approach: {loss_manual_fix.item():.4f}")
    print(f"\nProbability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
except Exception as e:
    print(f"⚠︎  Still getting an error: {e}")
    print(f"\nMake sure your probabilities are in the range [0, 1], and check [Path B] for possible input mismatches")