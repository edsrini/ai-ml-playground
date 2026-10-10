# Import core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datasets import load_from_disk, load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for visualizations
output_dir = 'assets/demo8'
os.makedirs(output_dir, exist_ok=True)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Setup complete!\n")

# 1. Load the dataset
print("Loading Forest Cover Type dataset...")
try:
    dataset = load_from_disk("/workspace/data/covertype-arrow")
except Exception as err:
    print(f"⚠︎ Local load failed ({err})). Will need to fall back to the Hugging Face Hub (may take 3-4 minutes)")
    dataset = load_dataset(
        "mstz/covertype",
        split="train",
        revision="refs/convert/parquet"
    )

df = dataset.to_pandas()
print(f"✓ Dataset loaded: {len(df):,} samples\n")

# 2. Subset data for faster training (still large enough for clear patterns)
subset_size = 20000
df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
print(f"✓ Using subset: {len(df):,} samples for faster demo\n")

# 3. Separate features and labels
# Labels are 1-7, convert to 0-6 for PyTorch
y = df['cover_type'].values
X = df.drop('cover_type', axis=1).values.astype(np.float64)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {df['cover_type'].nunique()}")
print(f"Class distribution:\n{df['cover_type'].value_counts().sort_index()}\n")

# 4. Normalize features (important for stable training)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
print("✓ Features normalized\n")

# 5. Split into train/val/test (50% train, 25% val, 25% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y, test_size=0.25, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp  # 0.333 of 75% = 25% of total
)

print(f"✓ Data split:")
print(f"  Training: {len(X_train):,} samples")
print(f"  Validation: {len(X_val):,} samples")
print(f"  Test: {len(X_test):,} samples\n")

# 6. Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# 7. Create DataLoaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ DataLoaders created with batch size {batch_size}")
print(f"  Training batches per epoch: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")


# Create the model class

class ForestClassifier(nn.Module):
    """Simple feedforward network for forest cover classification."""

    def __init__(self, input_size=54, hidden_size=64, num_classes=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Create three models with different capacities
model_underfit = ForestClassifier(hidden_size=4).to(device)
model_goodfit = ForestClassifier(hidden_size=64).to(device)
model_overfit = ForestClassifier(hidden_size=512).to(device)

print("✓ Three models created:")
print(f"  Underfit model: {sum(p.numel() for p in model_underfit.parameters()):,} parameters")
print(f"  Good fit model: {sum(p.numel() for p in model_goodfit.parameters()):,} parameters")
print(f"  Overfit model: {sum(p.numel() for p in model_overfit.parameters()):,} parameters")


# Define the training function
def train_and_track(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Train a model and track both training and validation losses.

    Returns:
        train_losses: List of average training loss per epoch
        val_losses: List of validation loss per epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        # Calculate average training loss for this epoch
        avg_train_loss = train_loss_sum / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_sum += loss.item()
                val_batches += 1

        # Calculate validation loss for this epoch
        avg_val_loss = val_loss_sum / val_batches
        val_losses.append(avg_val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


print("✓ Training function defined")

# Train the three models
epochs = 250
learning_rate = 0.001

print("=" * 60)
print("Training UNDERFIT model (4 hidden neurons)")
print("=" * 60)
train_losses_underfit, val_losses_underfit = train_and_track(
    model_underfit, train_loader, val_loader, epochs=epochs, lr=learning_rate
)

print("\n" + "=" * 60)
print("Training GOOD FIT model (64 hidden neurons)")
print("=" * 60)
train_losses_goodfit, val_losses_goodfit = train_and_track(
    model_goodfit, train_loader, val_loader, epochs=epochs, lr=learning_rate
)

print("\n" + "=" * 60)
print("Training OVERFIT model (512 hidden neurons)")
print("=" * 60)
train_losses_overfit, val_losses_overfit = train_and_track(
    model_overfit, train_loader, val_loader, epochs=epochs, lr=learning_rate
)

print("\n✓ All models trained!")

# Plot only training loss for the overfit model
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses_overfit, 'g-', linewidth=2, label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Only - Looks Perfect!', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'naive_view.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Final training loss: {train_losses_overfit[-1]:.4f}")
print("\n⚠︎  This looks amazing! Loss approaching zero = perfect model, right?")
print("⚠︎  WRONG. You can't tell if the model generalizes from training loss alone.")

# Plot both training and validation loss for the overfit model
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses_overfit, 'g-', linewidth=2, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses_overfit, 'r-', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss - The Truth Revealed', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'revealing_view.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Final training loss: {train_losses_overfit[-1]:.4f}")
print(f"Final validation loss: {val_losses_overfit[-1]:.4f}")
print(f"Gap: {val_losses_overfit[-1] - train_losses_overfit[-1]:.4f}")
print("\n⚠︎  The gap reveals the problem: model memorizes training data!")


