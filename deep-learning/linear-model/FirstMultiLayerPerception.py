# =========================
# 1. Import libraries
# =========================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

# Make random results repeatable
torch.manual_seed(42)
np.random.seed(42)

# Create folder for saving outputs later
output_dir = "assets/demo3"
os.makedirs(output_dir, exist_ok=True)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")


# =========================
# 2. Load restaurant dataset
# =========================

print("\nLoading restaurant dataset...")

dataset = load_dataset("MongoDB/whatscooking.restaurants", split="train")
df = pd.DataFrame(dataset)

print(f"Total restaurants loaded: {len(df):,}")
print("\nSample data:")
print(df[["name", "cuisine", "stars", "review_count", "borough"]].head())


# =========================
# 3. Keep only rows with required values
# =========================

required_columns = ["stars", "review_count", "PriceRange", "cuisine", "borough"]

df = df.dropna(subset=required_columns)

print(f"\nRestaurants after removing missing values: {len(df):,}")


# =========================
# 4. Create the target label
# =========================
# success = 1 if restaurant has:
#   - rating >= 4 stars
#   - review_count >= 50
# Otherwise success = 0

df["success"] = (
    (df["stars"] >= 4.0) &
    (df["review_count"] >= 50)
).astype(int)

y = df["success"].values


# =========================
# 5. Convert text columns into numbers
# =========================
# Neural networks need numbers, not text.
# Example: Italian → 0, Chinese → 1, etc.

cuisine_encoder = LabelEncoder()
borough_encoder = LabelEncoder()

df["cuisine_encoded"] = cuisine_encoder.fit_transform(df["cuisine"])
df["borough_encoded"] = borough_encoder.fit_transform(df["borough"])


# =========================
# 6. Select input features
# =========================
# These are the columns the model will use to make predictions.

feature_columns = [
    "stars",
    "review_count",
    "PriceRange",
    "cuisine_encoded",
    "borough_encoded"
]

X_raw = df[feature_columns].to_numpy()


# =========================
# 7. Scale input features
# =========================
# Scaling helps neural networks train better.
# It makes each feature centered around 0 with similar spread.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\nFeature matrix shape: {X_scaled.shape}")
print(f"Feature names: {feature_columns}")


# =========================
# 8. Convert data to PyTorch tensors
# =========================
# X_tensor = model inputs
# y_tensor = correct answers
#
# unsqueeze(1) changes y from shape (N,) to (N, 1)
# so it matches the model output shape.

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

print(f"\nX tensor shape: {X_tensor.shape}")
print(f"y tensor shape: {y_tensor.shape}")
print(f"Success rate: {df['success'].mean():.1%}")


# =========================
# 9. Build the neural network
# =========================

input_size = X_tensor.shape[1]   # Number of input features
hidden_size = 10                 # Number of hidden neurons
output_size = 1                  # One output: success probability

model_restaurant = nn.Sequential(
    nn.Linear(input_size, hidden_size),   # 5 features → 10 hidden neurons
    nn.Sigmoid(),                         # Adds non-linearity
    nn.Linear(hidden_size, output_size),  # 10 hidden neurons → 1 output
    nn.Sigmoid()                          # Converts output to probability 0–1
)

print("\nCreated restaurant success predictor!")
print(model_restaurant)

for name, param in model_restaurant.named_parameters():
    print(name)
    print(param[:2])  # print first few values

hidden_size=512
model_restaurant_wide = nn.Sequential(
    nn.Linear(input_size, hidden_size),   # 5 features → 10 hidden neurons
    nn.ReLU(),                         # Adds non-linearity
    nn.Linear(hidden_size, output_size),  # 10 hidden neurons → 1 output
    nn.Sigmoid()                          # Converts output to probability 0–1
)


# =========================
# 10. Count model parameters
# =========================

total_params = sum(p.numel() for p in model_restaurant.parameters())
print(f"\nTotal trainable parameters: {total_params}")

total_params = sum(p.numel() for p in model_restaurant_wide.parameters())
print(f"\nTotal trainable parameters wide: {total_params}")


# =========================
# 11. Make predictions before training
# =========================
# Important:
# This model has random weights right now.
# So predictions are just guesses.
#
# torch.no_grad() means:
# "Only predict. Do not track gradients."

sample_size = 10
sample_X = X_tensor[:sample_size]
sample_y = y_tensor[:sample_size]

with torch.no_grad():
    predictions = model_restaurant(sample_X)
    predicted_classes = (predictions > 0.5).float()


# =========================
# 13. Print prediction results
# =========================

print("\nPredictions with random weights:\n")
print("Restaurant | Actual Success | Predicted Probability | Predicted Class")
print("-" * 75)

for i in range(sample_size):
    actual = sample_y[i].item()
    restaurant = sample_X[i]
    pred_prob = predictions[i].item()
    pred_class = predicted_classes[i].item()

    print(f"{i:10d} | {actual:14.0f} | {pred_prob:21.4f} | {pred_class:15.0f}")

accuracy = (predicted_classes == sample_y).float().mean().item()

print(f"\nSample accuracy: {accuracy:.1%}")
print(f"Model predicts success: {predicted_classes.mean().item():.0%} of the time")
print(f"Actual success rate in sample: {sample_y.mean().item():.0%}")
print("\nNote: The model is not trained yet, so these predictions are random guesses.")

results = pd.DataFrame({
    "name": df.iloc[:sample_size]["name"].values,
    "cuisine": df.iloc[:sample_size]["cuisine"].values,
    "stars": df.iloc[:sample_size]["stars"].values,
    "review_count": df.iloc[:sample_size]["review_count"].values,
    "actual_success": sample_y.squeeze().numpy(),
    "predicted_probability": predictions.squeeze().numpy(),
    "predicted_class": predicted_classes.squeeze().numpy()
})

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(results.round(3).to_string(index=False))
print("\nPredictions with input details:\n")
print(results)