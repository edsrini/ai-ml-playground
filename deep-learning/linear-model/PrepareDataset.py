# Import core libraries
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")

# 1. Load the dataset from Hugging Face
dataset = load_dataset('aai510-group1/telco-customer-churn', split='train')

# Convert to Pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

print(f"✓ Dataset loaded: {len(df)} customer records\n")

# 2. Select the 10 key features we'll use for prediction
feature_columns = [
    'Age',                          # Demographic
    'Dependents',                   # Demographic
    'Tenure in Months',             # Service history
    'Contract',                     # Service details (categorical)
    'Internet Type',                # Service details (categorical - has missing values!)
    'Monthly Charge',               # Financial
    'Total Charges',                # Financial
    'Satisfaction Score',           # Engagement
    'Number of Referrals',          # Engagement
]

target_column = 'Churn'  # Binary: 0 = Stayed, 1 = Churned

# Keep only the columns we need
df = df[feature_columns + [target_column]]

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check data types and missing values
print("Data Types and Missing Values:")
print("=" * 60)
info_df = pd.DataFrame({
    'Column': df.columns,
    'Type': df.dtypes.values,
    'Missing': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
})
print(info_df.to_string(index=False))

# Handle missing values in Internet Type
# Fill with "None" to indicate "no internet service"
df['Internet Type'] = df['Internet Type'].fillna('None')

# Verify no missing values remain
print("Missing values after handling:")
print(df.isnull().sum())
print("\n✓ All missing values handled!")

# Check unique values in categorical columns
print("Categorical features:")
print(f"Contract types: {df['Contract'].unique()}")
print(f"Internet types: {df['Internet Type'].unique()}")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Contract', 'Internet Type'], drop_first=True)

print(f"\n✓ Encoding complete!")
print(f"Original shape: {df.shape}")
print(f"After encoding: {df_encoded.shape}")
print(f"\nNew columns created: {df_encoded.shape[1] - df.shape[1]} additional features")

# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print("Before split:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split into training (70%) and validation (30%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.30,      # 30% for validation
    random_state=42,     # For reproducibility
    stratify=y           # Maintain the same churn ratio in both sets
)

print("\n" + "=" * 60)
print("After split:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"\nTraining target distribution: {y_train.value_counts().to_dict()}")
print(f"Validation target distribution: {y_val.value_counts().to_dict()}")
print("\n✓ Data split complete! (Golden rule: Split BEFORE preprocessing)")

# Check the current feature scales
print("Feature scales BEFORE standardization:")
print("=" * 60)
print(X_train.describe().loc[['mean', 'std', 'min', 'max']].round(2))

# Initialize the scaler
scaler = StandardScaler()

# CRITICAL: Fit the scaler ONLY on training data
# This calculates mean and std from training set only
scaler.fit(X_train)

# Transform both sets using the training statistics
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n✓ Standardization complete!")
print("\nFeature scales AFTER standardization (training set):")
print("=" * 60)
# Convert back to DataFrame for easier viewing
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
print(X_train_scaled_df.describe().loc[['mean', 'std', 'min', 'max']].round(2))

# Convert features to float32 tensors (standard for neural networks)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# Convert targets to float32 tensors (for binary classification with BCELoss)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

print("✓ Tensor conversion complete!")
print("\nTraining tensors:")
print(f"  Features: {X_train_tensor.shape} | dtype: {X_train_tensor.dtype}")
print(f"  Targets:  {y_train_tensor.shape} | dtype: {y_train_tensor.dtype}")
print("\nValidation tensors:")
print(f"  Features: {X_val_tensor.shape} | dtype: {X_val_tensor.dtype}")
print(f"  Targets:  {y_val_tensor.shape} | dtype: {y_val_tensor.dtype}")

# Quick sanity check: print first example
print("\nFirst training example:")
print(f"Features (first 5): {X_train_tensor[0, :5]}")
print(f"Target: {y_train_tensor[0].item()} ({'Churned' if y_train_tensor[0].item() == 1 else 'Stayed'})")

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

print("✓ TensorDatasets created!")
print(f"\nTraining dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")

# Show what a single dataset item looks like
features, target = train_dataset[0]
print("\nSingle dataset item:")
print(f"  Features shape: {features.shape}")
print(f"  Target shape: {target.shape}")
print(f"  Target value: {target.item()}")

# Create DataLoaders
batch_size = 32  # Process 32 examples at a time

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,          # Shuffle training data each epoch
    drop_last=False        # Keep the last incomplete batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,         # Don't shuffle validation data
    drop_last=False
)

print("✓ DataLoaders created!")
print(f"\nTraining loader:")
print(f"  Total examples: {len(train_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(train_loader)}")
print(f"  Shuffle: True")

print(f"\nValidation loader:")
print(f"  Total examples: {len(val_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(val_loader)}")
print(f"  Shuffle: False")

