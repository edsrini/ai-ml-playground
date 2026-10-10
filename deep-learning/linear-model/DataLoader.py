# Import core libraries
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datasets import load_dataset
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")

# 1. Load dataset
dataset = load_dataset('aai510-group1/telco-customer-churn', split='train')
df = pd.DataFrame(dataset)

# 2. Select features
feature_columns = [
    'Age', 'Dependents', 'Tenure in Months', 'Contract', 'Internet Type',
    'Monthly Charge', 'Total Charges', 'Satisfaction Score', 'Number of Referrals'
]
target_column = 'Churn'
df = df[feature_columns + [target_column]]

# 3. Handle missing values
df['Internet Type'] = df['Internet Type'].fillna('None')

# 4. Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Contract', 'Internet Type'], drop_first=True)

# 5. Split data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"✓ Data loaded and preprocessed")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")


# Build the reusable scaling pipeline

def create_scaled_pipeline(X_train, X_val, y_train, y_val, scaler, batch_size=32):

    X_train_scaled = scaler.fit_transform(X_train)  
    X_val_scaled = scaler.transform(X_val)  

    # TODO: Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)  
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)  
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)  

    # TODO: Create TensorDatasets to pair features with targets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  

    # TODO: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )  

    return train_loader, val_loader


# Standardize the features using StandardScaler

# TODO: Initialize StandardScaler
standard_scaler = StandardScaler() 

# Create StandardScaler pipeline
baseline_train_loader, baseline_val_loader = create_scaled_pipeline(
    X_train, X_val, y_train, y_val,
    scaler=standard_scaler,
    batch_size=32
)

print("✓ StandardScaler pipeline created!")
print(f"  Batch size: 32")
print(f"  Number of training batches: {len(baseline_train_loader)}")
print(f"  Number of validation batches: {len(baseline_val_loader)}")

# Inspect baseline data values in a batch to understand what StandardScaler produces and why it matters for training.

# TODO: Get one batch from the baseline DataLoader
batch_features, batch_targets = next(iter(baseline_train_loader))

# Examine the value range for the batch
print("StandardScaler feature statistics:")
print("=" * 60)
print(f"Mean across batch: {batch_features.mean(dim=0)[:5]}")
print(f"Std across batch:  {batch_features.std(dim=0)[:5]}")
print(f"Min value in batch: {batch_features.min().item():.3f}")
print(f"Max value in batch: {batch_features.max().item():.3f}")
print(f"\nFirst example (first 5 features): {batch_features[0, :5]}")

batch_df = pd.DataFrame(
    batch_features.numpy(),
    columns=X_train.columns
)

print(batch_df)

#Standardize the features using MinMaxScaler

# TODO: Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Create MinMaxScaler pipeline
normalized_train_loader, normalized_val_loader = create_scaled_pipeline(
    X_train, X_val, y_train, y_val,
    scaler=minmax_scaler,
    batch_size=32
)

print("✓ MinMaxScaler pipeline created!")
print(f"  Batch size: 32")
print(f"  Number of training batches: {len(normalized_train_loader)}")
print(f"  Number of validation batches: {len(normalized_val_loader)}")

# Check device - FORCE CPU to avoid CUDA compatibility error
device = torch.device("cpu")
print(f"Using device: {device}")


def test_batch_size(batch_size, loader, device=device, num_warmup=2, num_runs=3):
    times = []

    # Get actual number of input features dynamically
    input_size = loader.dataset.tensors[0].shape[1]

    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).to(device)

    dummy_model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                _ = dummy_model(batch_features)

        # Timed runs
        for _ in range(num_runs):
            start = time.time()

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                _ = dummy_model(batch_features)

            iteration_time = time.time() - start
            times.append(iteration_time)

    return np.mean(times), np.std(times)

# Create DataLoaders with different batch sizes and measure iteration time

# TODO: Choose your batch sizes
batch_sizes = [8, 32, 64]  # Add your choices here

# Test different batch sizes
results = {}
print("Testing different batch sizes:")
print("=" * 60)

for bs in batch_sizes:
    # TODO: Create DataLoader with specified batch_size
    scaler = StandardScaler()
    train_loader, val_loader = create_scaled_pipeline(
        X_train, X_val, y_train, y_val,
        scaler=scaler,
        batch_size=bs
    )

    # Time both loaders
    train_mean, train_std = test_batch_size(bs, train_loader, device)
    val_mean, val_std = test_batch_size(bs, val_loader, device)

    results[bs] = {
        'train_time': train_mean,
        'train_std': train_std,
        'val_time': val_mean,
        'val_std': val_std,
        'num_train_batches': len(train_loader),
        'num_val_batches': len(val_loader),
    }

    print(f"\nBatch size {bs}:")
    print(f"  Training batches: {results[bs]['num_train_batches']}")
    print(f"  Training time: {results[bs]['train_time']:.4f} ± {results[bs]['train_std']:.4f} seconds")
    print(f"  Time per batch: {results[bs]['train_time'] / len(train_loader) * 1000:.2f} ms/batch")
    print(f"  Validation batches: {results[bs]['num_val_batches']}")
    print(f"  Validation time: {results[bs]['val_time']:.4f} ± {results[bs]['val_std']:.4f} seconds")
    print(f"  Time per batch: {results[bs]['val_time'] / len(val_loader) * 1000:.2f} ms/batch")


