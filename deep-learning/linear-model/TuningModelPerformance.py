# Import core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create output directory for visualizations
output_dir = 'assets/model_improvements'
os.makedirs(output_dir, exist_ok=True)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Setup complete!\n")

# 1. Load the dataset
dataset = load_dataset('mstz/wine', split='train')
df = pd.DataFrame(dataset)

print(f"✓ Dataset loaded: {len(df)} wines")
print(f"✓ Features: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# 2. Create binary classification labels
# High quality: score >= 7, Low quality: score < 7
df['high_quality'] = (df['quality'] >= 7).astype(int)

# Check class distribution
class_counts = df['high_quality'].value_counts()
print(f"\n✓ Binary classification created:")
print(f"   Low quality (0): {class_counts[0]} wines ({class_counts[0]/len(df)*100:.1f}%)")
print(f"   High quality (1): {class_counts[1]} wines ({class_counts[1]/len(df)*100:.1f}%)")

# 3. Separate features and target
feature_columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'is_red']

X = df[feature_columns].values.astype(np.float32)
y = df['high_quality'].values.astype(np.float32)

# 4. Normalize features (important for neural networks!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\n✓ Features normalized to zero mean and unit variance")
print(f"✓ Final dataset shape: {X.shape}")

# 5. Create train and validation splits with stratification to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Create PyTorch dataset
class WineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create dataset and split into train/validation
train_dataset = WineDataset(X_train, y_train)
val_dataset = WineDataset(X_val, y_val)

# 6. Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Validation samples: {len(val_dataset)}")
print(f"✓ Batch size: {batch_size}")


# Baseline model WITHOUT regularization
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BaselineModel, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Create baseline model
input_size = X.shape[1]  # 12 features
hidden_sizes = [128, 64, 32]  # Three hidden layers
output_size = 1  # Binary classification

baseline_model = BaselineModel(input_size, hidden_sizes, output_size).to(device)
print("✓ Baseline model created (NO regularization)")
print(f"  Architecture: {input_size} → 128 → 64 → 32 → {output_size}")
print(f"  Total parameters: {sum(p.numel() for p in baseline_model.parameters())}")


def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs=50, scheduler=None, model_name="Model"):
    """
    Train a model and track training/validation metrics.
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.0).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (outputs > 0.0).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")

    return train_losses, val_losses, train_accs, val_accs


baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("✓ Training function defined, with criterion and optimizer")

# Train baseline model
print("Training baseline model (no regularization)...\n")
epochs = 100

baseline_train_losses, baseline_val_losses, baseline_train_accs, baseline_val_accs = train_model(
    baseline_model, train_loader, val_loader, baseline_optimizer, criterion,
    epochs=epochs, model_name="Baseline"
)

print(f"\n✓ Baseline training complete!")
print(f"  Final train accuracy: {baseline_train_accs[-1]:.2f}%")
print(f"  Final validation accuracy: {baseline_val_accs[-1]:.2f}%")
print(f"  Gap: {baseline_train_accs[-1] - baseline_val_accs[-1]:.2f}%)")

# Visualize baseline performance
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
axes[0].plot(baseline_train_losses, label='Train Loss', linewidth=2)
axes[0].plot(baseline_val_losses, label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Baseline Model: Loss Curves\n(Notice the growing gap!)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(baseline_train_accs, label='Train Accuracy', linewidth=2)
axes[1].plot(baseline_val_accs, label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Baseline Model: Accuracy Curves', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'baseline_performance.png'), dpi=300, bbox_inches='tight')
#plt.show()


# Model WITH dropout
class ModelWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ModelWithDropout, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Add dropout!

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout after each hidden layer!

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Create model with dropout
dropout_model = ModelWithDropout(input_size, hidden_sizes, output_size, dropout_rate=0.3).to(device)
print("✓ Model with dropout created")
print(f"  Architecture: {input_size} → 128 (+dropout) → 64 (+dropout) → 32 (+dropout) → {output_size}")
print(f"  Dropout rate: 30% of neurons randomly disabled during training")

# Demonstrate dropout randomness
dropout_model.train()  # Enable training mode (dropout is active)

# Get a sample input
sample_input = train_dataset[0][0].unsqueeze(0).to(device)

# Run the same input through the network three times
with torch.no_grad():
    output1 = dropout_model(sample_input).item()
    output2 = dropout_model(sample_input).item()
    output3 = dropout_model(sample_input).item()

print("Same input, three forward passes in TRAIN mode (dropout active):")
print(f"  Pass 1: {output1:.6f}")
print(f"  Pass 2: {output2:.6f}")
print(f"  Pass 3: {output3:.6f}")
print(f"  → Outputs differ because different neurons are dropped each time!\n")

# Now try in eval mode
dropout_model.eval()  # Disable dropout

with torch.no_grad():
    output1 = dropout_model(sample_input).item()
    output2 = dropout_model(sample_input).item()
    output3 = dropout_model(sample_input).item()

print("Same input, three forward passes in EVAL mode (dropout disabled):")
print(f"  Pass 1: {output1:.6f}")
print(f"  Pass 2: {output2:.6f}")
print(f"  Pass 3: {output3:.6f}")
print(f"  → Outputs are identical because all neurons are active!")

# Train model with dropout
print("Training model WITH dropout...\n")

dropout_optimizer = optim.Adam(dropout_model.parameters(), lr=0.001)

dropout_train_losses, dropout_val_losses, dropout_train_accs, dropout_val_accs = train_model(
    dropout_model, train_loader, val_loader, dropout_optimizer, criterion,
    epochs=epochs, model_name="Dropout"
)

print(f"\n✓ Dropout model training complete!")
print(f"  Final train accuracy: {dropout_train_accs[-1]:.2f}%")
print(f"  Final validation accuracy: {dropout_val_accs[-1]:.2f}%")
print(f"  Gap: {dropout_train_accs[-1] - dropout_val_accs[-1]:.2f}%")


# Create model with dropout AND learning rate scheduling
scheduled_model = ModelWithDropout(input_size, hidden_sizes, output_size, dropout_rate=0.3).to(device)
scheduled_optimizer = optim.Adam(scheduled_model.parameters(), lr=0.001)

# Add learning rate scheduler: reduce LR by 0.5 every 15 epochs
scheduler = optim.lr_scheduler.StepLR(scheduled_optimizer, step_size=15, gamma=0.5)

print("✓ Model with dropout + LR scheduling created")
print(f"  Starting learning rate: {scheduled_optimizer.param_groups[0]['lr']}")
print(f"  LR will decay by 0.5× every 15 epochs")
print(f"  Example: 0.001 → 0.0005 (epoch 15) → 0.00025 (epoch 30) → 0.000125 (epoch 45)")

# Train with dropout + scheduling
print("\nTraining model WITH dropout + LR scheduling...\n")

scheduled_train_losses, scheduled_val_losses, scheduled_train_accs, scheduled_val_accs = train_model(
    scheduled_model, train_loader, val_loader, scheduled_optimizer, criterion,
    epochs=epochs, scheduler=scheduler, model_name="Dropout+Scheduling"
)

print(f"\n✓ Scheduled model training complete!")
print(f"  Final train accuracy: {scheduled_train_accs[-1]:.2f}%")
print(f"  Final validation accuracy: {scheduled_val_accs[-1]:.2f}%")
print(f"  Gap: {scheduled_train_accs[-1] - scheduled_val_accs[-1]:.2f}%")

torch.save(scheduled_model.state_dict(), os.path.join(output_dir, 'model_wine.pth'))