# Import core libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
output_dir = 'assets/demo9'
os.makedirs(output_dir, exist_ok=True)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Setup complete!\n")

# 1. Load the dataset
print("Loading credit card fraud dataset...")
dataset = load_dataset('David-Egea/Creditcard-fraud-detection', split='train')

# Convert to pandas DataFrame for easier manipulation
df = dataset.to_pandas()

# 2. Subset to 50,000 transactions for manageable demo
df = df.sample(n=50000, random_state=42).reset_index(drop=True)


print(f"✓ Dataset loaded: {len(df)} transactions")
print(f"✓ Features: {df.shape[1] - 1} (V1-V28, Time, Amount)")
print(f"✓ Target: Class (0=legitimate, 1=fraud)\n")

# 3. Explore the class imbalance
fraud_count = (df['Class'] == 1).sum()
legit_count = (df['Class'] == 0).sum()
fraud_percentage = (fraud_count / len(df)) * 100

print("="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(f"Legitimate transactions: {legit_count:,} ({100-fraud_percentage:.3f}%)")
print(f"Fraudulent transactions: {fraud_count:,} ({fraud_percentage:.3f}%)")
print(f"\nImbalance ratio: {legit_count/fraud_count:.0f}:1")
print("="*60)

# Visualize the imbalance
fig, ax = plt.subplots(figsize=(8, 6))
counts = [legit_count, fraud_count]
labels = [f'Legitimate\n{legit_count:,}\n({100-fraud_percentage:.3f}%)',
          f'Fraud\n{fraud_count:,}\n({fraud_percentage:.3f}%)']
colors = ['#90EE90', '#FF6B6B']

ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Transactions', fontsize=12)
ax.set_title('Severe Class Imbalance in Fraud Detection', fontsize=14, fontweight='bold')
ax.set_yscale('log')  # Log scale to show both classes
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_imbalance.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Split features from labels
X = df.drop('Class', axis=1).values  # All features (V1-V28, Time, Amount)
y = df['Class'].values  # Binary labels (0 or 1)

print(f"✓ Features shape: {X.shape}")
print(f"✓ Labels shape: {y.shape}")

# 5. Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"    ✓ Train fraud rate: {(y_train.sum()/len(y_train))*100:.3f}%")
print(f"    ✓ Test fraud rate: {(y_test.sum()/len(y_test))*100:.3f}%")

# 6. Normalize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features normalized (mean=0, std=1) using training set statistics")

# 7. Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

print(f"✓ Converted to PyTorch tensors")
print(f"  - X_train: {X_train_tensor.shape}")
print(f"  - X_test: {X_test_tensor.shape}")
print("\nData preparation complete!\n")


# 1. Define the model architecture
class FraudDetector(nn.Module):
    def __init__(self, input_size=30):
        super(FraudDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Return raw logits (no sigmoid)
        return x


# 2. Create model and load pre-trained weights
model = FraudDetector(input_size=30).to(device)
model_path = os.path.join(output_dir, 'fraud_detector.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✓ Pre-trained model loaded successfully")
else:
    print("No saved model found. Training new model...")

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        logits = model(X_train_tensor).squeeze()
        loss = criterion(logits, y_train_tensor)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✓ Model trained and saved to: {model_path}")

# 3. Set model to evaluation mode
model.eval()

print("✓ Pre-trained model loaded successfully")

# Get predictions from the model
with torch.no_grad():
    # Get raw logits from model
    logits = model(X_test_tensor.to(device)).squeeze()

    # Convert logits to probabilities using sigmoid (since we used BCEWithLogitsLoss for training)
    probabilities = torch.sigmoid(logits).cpu()

    # Convert probabilities to binary predictions using 0.5 threshold
    predictions = (probabilities >= 0.5).float()
