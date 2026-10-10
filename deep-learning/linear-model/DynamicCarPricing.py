# Import core libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
import warnings

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for visualizations
output_dir = 'assets/demo6'
os.makedirs(output_dir, exist_ok=True)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Setup complete!\n")

# 1. Load the dataset from Hugging Face
dataset = load_dataset('Vishaltiwari2019/Car-Price-Prediction', split='train')
df = pd.DataFrame(dataset)

print(f"✓ Dataset loaded: {len(df)} cars found\n")
print("Sample of the data:")
print(df.head())
print(f"\nPrice range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")

# 2. Encode categorical variables
label_encoders = {}
categorical_columns = ['Make', 'Model', 'Fuel Type', 'Transmission']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✓ Categorical variables encoded")
print(f"\nEncoded data sample:")
print(df.head())

# 3. Separate features and target
X = df.drop('Price', axis=1).values
y = df['Price'].values.reshape(-1, 1)

# 4. Normalize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

print(f"\nScaling data sample:")

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)


# 5. Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

print(f"\n✓ Data prepared for training")
print(f"Total samples: {len(X)}")
print(f"Input features: {X.shape[1]}")


class PricePredictionModel(nn.Module):
    """
    A simple feedforward neural network for regression.

    Architecture: 7 inputs -> 64 neurons -> 32 neurons -> 1 output
    """

    def __init__(self, input_size=7):
        super(PricePredictionModel, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output for regression
        return x


# Create the model
model = PricePredictionModel(input_size=7)
print("✓ Model created")
print(f"\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Make predictions with untrained model (on first 200 samples)
with torch.no_grad():
    initial_predictions = model(X_tensor[:200]).numpy()

# Convert back to original price scale
initial_predictions_orig = scaler_y.inverse_transform(initial_predictions)
y_orig = scaler_y.inverse_transform(y[:200])


print("Untrained model predictions (first 5 examples):")
print("-" * 50)
for i in range(5):
    print(
        f"Actual: ${y_orig[i, 0]:>8,.2f}  |  Predicted: ${initial_predictions_orig[i, 0]:>8,.2f}  |  Error: ${abs(y_orig[i, 0] - initial_predictions_orig[i, 0]):>8,.2f}")

# Calculate average error
avg_error = np.mean(np.abs(y_orig - initial_predictions_orig))
print(f"\nAverage absolute error: ${avg_error:,.2f}")
print("\n✖ The model is basically guessing! Let's train it...")


# Set up training components
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training components:")
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Learning rate: 0.001")

# Let's examine one specific weight before training
# We'll track the first weight in the first layer
initial_weight = model.fc1.weight[0, 0].item()
print(f"\nTracking one weight before training:")
print(f"model.fc1.weight[0, 0] = {initial_weight:.6f}")

print("=" * 60)
print("EXECUTING ONE TRAINING STEP")
print("=" * 60)

# ============================================================
# STEP 1: Forward Pass
# ============================================================
print("\n STEP 1: Forward Pass")
print("-" * 60)
predictions = model(X_tensor)
print(f"Input shape: {X_tensor.shape}")
print(f"Output shape: {predictions.shape}")
print(f"Sample prediction: {predictions[0].item():.4f}")
print("✓ Data passed through network, predictions generated")

# ============================================================
# STEP 2: Compute Loss
# ============================================================
print("\n STEP 2: Compute Loss")
print("-" * 60)
loss = criterion(predictions, y_tensor)
print(f"Loss value: {loss.item():.4f}")
print("✓ Measured how wrong predictions are using MSE")

# ============================================================
# STEP 3: Zero Gradients
# ============================================================
print("\n STEP 3: Zero Gradients")
print("-" * 60)
# Check gradient before zeroing (will be None on first iteration)
grad_before = model.fc1.weight.grad
print(f"Gradient before zeroing: {grad_before}")

optimizer.zero_grad()
grad_after = model.fc1.weight.grad
print(f"Gradient after zeroing: {grad_after}")
print("✓ Cleared old gradients to prepare for new computation")

# ============================================================
# STEP 4: Backward Pass (Backpropagation)
# ============================================================
print("\n STEP 4: Backward Pass")
print("-" * 60)
loss.backward()
# Now gradients are computed
sample_gradient = model.fc1.weight.grad[0, 0].item()
print(f"Sample gradient (model.fc1.weight[0, 0]): {sample_gradient:.6f}")
print("✓ Computed gradients for all parameters using backpropagation")

# ============================================================
# STEP 5: Update Weights
# ============================================================
print("\n STEP 5: Update Weights")
print("-" * 60)
weight_before_update = model.fc1.weight[0, 0].item()
print(f"Weight before update: {weight_before_update:.6f}")

optimizer.step()

weight_after_update = model.fc1.weight[0, 0].item()
weight_change = weight_after_update - weight_before_update
print(f"Weight after update:  {weight_after_update:.6f}")
print(f"Change in weight:     {weight_change:.6f}")
print("✓ Applied gradients to update all weights")

print("\n" + "=" * 60)
print("ONE TRAINING STEP COMPLETE!")
print("=" * 60)

# Reset the model for clean training
model = PricePredictionModel(input_size=7)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track training progress
num_epochs = 100
train_losses = []

print("Training the model...\n")
print(f"{'Epoch':>6} | {'Training Loss':>14}")
print("-" * 25)

for epoch in range(num_epochs):
    # The 5-step training loop
    predictions = model(X_tensor)  # Step 1: Forward pass
    loss = criterion(predictions, y_tensor)  # Step 2: Compute loss
    optimizer.zero_grad()  # Step 3: Zero gradients
    loss.backward()  # Step 4: Backward pass
    optimizer.step()  # Step 5: Update weights

    train_losses.append(loss.item())

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"{epoch + 1:6d} | {loss.item():14.4f}")

print("\n✓ Training complete!")
# Get trained model predictions (on same 200 samples we evaluated initially)
with torch.no_grad():
    trained_predictions = model(X_tensor[:200]).numpy()

# Convert back to original price scale
trained_predictions_orig = scaler_y.inverse_transform(trained_predictions)

print("Before vs. After Training (first 10 examples):")
print("=" * 70)
print(f"{'Actual Price':>15} | {'Initial Pred':>15} | {'Trained Pred':>15}")
print("-" * 70)

for i in range(10):
    actual = y_orig[i, 0]
    initial_pred = initial_predictions_orig[i, 0]
    trained_pred = trained_predictions_orig[i, 0]

    print(f"${actual:>13,.2f} | ${initial_pred:>13,.2f} | ${trained_pred:>13,.2f}")

# Calculate overall improvement
initial_mae = np.mean(np.abs(y_orig - initial_predictions_orig))
trained_mae = np.mean(np.abs(y_orig - trained_predictions_orig))
improvement_pct = (initial_mae - trained_mae) / initial_mae * 100

print("\n" + "=" * 70)
print(f"Average absolute error (before): ${initial_mae:>10,.2f}")
print(f"Average absolute error (after):  ${trained_mae:>10,.2f}")
print(f"Error reduction:                  {improvement_pct:>9.1f}%")
print("=" * 70)
