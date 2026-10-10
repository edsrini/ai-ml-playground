import torch

# -------------------------
# 1. Input data (AND gate)
# -------------------------
# Each row = [input1, input2]
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

# Expected output for AND gate
y = torch.tensor([0., 0., 0., 1.])

# -------------------------
# 2. Create PerceptronANDGate.py
# -------------------------
class Perceptron:
    def __init__(self):
        # weights (importance of each input)
        self.w = torch.randn(2)

        # bias (threshold shift)
        self.b = torch.randn(1)

    def forward(self, X):
        # Step 1: multiply inputs with weights and add bias
        z = torch.matmul(X, self.w) + self.b

        # Step 2: apply step function
        # if z > 0 → 1 else → 0
        return (z > 0).float()

    def accuracy(self, X, y):
        # get predictions
        preds = self.forward(X)

        # compare with actual answers
        # mean() gives % correct
        return (preds == y).float().mean().item()

# -------------------------
# 3. Create model
# -------------------------
model = Perceptron()

# -------------------------
# 4. Manually set weights
# (this solves AND gate)
# -------------------------
model.w = torch.tensor([1.0, 1.0])
model.b = torch.tensor([-1.5])

# -------------------------
# 5. Run model
# -------------------------
preds = model.forward(X)

print("Predictions:", preds)
print("Accuracy:", model.accuracy(X, y))