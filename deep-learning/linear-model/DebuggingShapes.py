# Import core libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for visualizations
output_dir = 'assets/demo4'
os.makedirs(output_dir, exist_ok=True)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Setup complete!\n")

# 1. Load the dataset
dataset = load_dataset('Redsmoothy/HR_Attrition', split='train')
print(f"✓ Dataset loaded: {len(dataset)} employees found\n")

# 2. Convert to pandas for easier preprocessing
df = pd.DataFrame(dataset)

# Display first few rows
print("Sample of raw data:")
print(df[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'OverTime', 'Attrition']].head())
print(f"\nTarget distribution: {df['Attrition'].value_counts().to_dict()}")

# 3. Select 10 key numeric features for predicting attrition
feature_columns = [
    'Age',                     # Career stage indicator
    'DistanceFromHome',        # Commute burden
    'MonthlyIncome',           # Compensation level
    'TotalWorkingYears',       # Experience level
    'YearsAtCompany',          # Tenure
    'YearsInCurrentRole',      # Role stagnation indicator
    'YearsSinceLastPromotion', # Career growth indicator
    'YearsWithCurrManager',    # Management relationship
    'WorkLifeBalance',         # Work-life satisfaction (1-4 rating)
    'OverTime'                 # Work intensity (will encode as binary)
]

# Create feature dataframe
features_df = df[feature_columns].copy()

# 4. Encode OverTime: Yes→1, No→0
features_df['OverTime'] = (features_df['OverTime'] == 'Yes').astype(int)

print("✓ Selected 10 key features for the model")
print("\nFeature statistics:")
pd.set_option('display.max_columns', None)
print(features_df.describe())

# 5. Normalize features to [0, 1] range
# Store min/max for each feature for normalization
feature_mins = features_df.min()
feature_maxs = features_df.max()

# Normalize each feature
scaler = MinMaxScaler()

features_normalized = scaler.fit_transform(features_df)

features_normalized = pd.DataFrame(
    features_normalized,
    columns=features_df.columns
)

print("✓ Features normalized to [0, 1] range")
print("\nNormalized feature ranges:")
print(f"Min: {features_normalized.min().max():.4f}")
print(f"Max: {features_normalized.max().min():.4f}")

print(features_normalized.describe())

# 6. Encode target variable: Attrition (Yes→1, No→0)
target = (df['Attrition'] == 'Yes').astype(int).values

print(f"✓ Target encoded: Yes→1 (will leave), No→0 (will stay)")
print(f"\nAttrition rate: {target.mean():.1%} of employees left")

# 7. Convert to PyTorch tensors
X = torch.FloatTensor(features_normalized.values)
y = torch.FloatTensor(target).unsqueeze(1)  # Add dimension: (N,) → (N, 1) to match the network's output dimensions

print(f"✓ Converted to PyTorch tensors")
print(f"\nFinal shapes:")
print(f"  Features (X): {X.shape} → {X.shape[0]} employees, {X.shape[1]} features each")
print(f"  Target (y):   {y.shape} → {y.shape[0]} labels (1=will leave, 0=will stay)")
print(f"\nDataset ready for training!")

# Build a simple model with nn.Sequential
sequential_model = nn.Sequential(
    nn.Linear(10, 20),  # Input: 10 features → Output: 20 features
    nn.ReLU(),          # Activation function
    nn.Linear(20, 1),   # Output: 1 prediction (attrition probability)
    nn.Sigmoid()        # Convert to probability [0, 1]
)

print("Sequential Model Architecture:")
print(sequential_model)
print("\n" + "="*60)

# Take a small batch of 5 employees and run through the model
sample_batch = X[:5]  # First 5 employees

print(f"Input shape: {sample_batch.shape}")
print("Running through sequential_model...\n")

# Forward pass
output = sequential_model(sample_batch)

print(f"Output shape: {output.shape}")
print(f"\nPredictions (attrition probabilities):")
for i, prob in enumerate(output.squeeze()):
    print(f"  Employee {i+1}: {prob.item():.1%} risk of leaving")


class AttritionMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for predicting employee attrition.

    Architecture:
    - Input layer: 10 features (employee characteristics)
    - Hidden layer: 20 neurons with ReLU activation
    - Output layer: 1 neuron with Sigmoid activation (attrition probability)
    """

    def __init__(self):
        """
        Initialize the network layers.
        This is where we DEFINE what layers exist in our network.
        """
        super(AttritionMLP, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(10, 20)  # First transformation: 10 → 20 features
        self.layer2 = nn.Linear(20, 1)  # Second transformation: 20 → 1 output

    def forward(self, x):
        """
        Define the forward pass: how data flows through the network.
        This is where we SPECIFY the order and transformations.

        Args:
            x: Input tensor of shape (batch_size, 10)

        Returns:
            Output tensor of shape (batch_size, 1) with attrition probabilities
        """
        # Print input shape
        print(f"Input shape: {x.shape}")

        # First layer: linear transformation
        x = self.layer1(x)
        print(f"After layer1 (Linear 10→20): {x.shape}")

        # Apply ReLU activation
        x = torch.relu(x)
        print(f"After ReLU: {x.shape}")

        # Second layer: linear transformation
        x = self.layer2(x)
        print(f"After layer2 (Linear 20→1): {x.shape}")

        # Apply sigmoid to get probability
        x = torch.sigmoid(x)
        print(f"After Sigmoid: {x.shape}")

        return x


# Instantiate the model
model = AttritionMLP()

print("\nCustom Model Architecture:")
print(model)
print("\n" + "=" * 60)

# Use the same 5-employee sample
print("Running forward pass with shape tracking:\n")
print("="*60)

# Forward pass - this automatically calls model.forward(sample_batch)
output = model(sample_batch)

print("="*60)
print(f"\nFinal predictions (attrition probabilities):")
for i, prob in enumerate(output.squeeze()):
    print(f"  Employee {i+1}: {prob.item():.1%} risk of leaving")