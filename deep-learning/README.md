# Deep Learning

PyTorch exercises, from a single neuron up to multi-layer networks, plus a small GPT trained from scratch.

| Folder | What's in it |
|---|---|
| [`linear-function`](linear-function) | First steps: logic-gate learning (`ANGGate.py`) and a small regression (`MoviePromotion.py`) with `torch.nn` |
| [`linear-model`](linear-model) | The main PyTorch exercises: logic gates (AND, XOR), a first multi-layer perceptron, loss functions, datasets and `DataLoader`, training and validation loops, evaluation strategy, tuning model performance, debugging tensor shapes, z-score normalization, and applied examples (`DynamicCarPricing.py`, `GreenHouse.py`) |
| [`tiny-gpt`](tiny-gpt) | A character-level GPT (about 10M parameters) built and trained from scratch on Tiny Shakespeare. Not yet run end to end |

## Running the exercises

Each script is standalone: run it from inside its folder (`python ANDGate.py`). They use `torch`, `numpy` and `pandas`; `linear-model/test_ComputeLaunch.py` is a small `pytest` example. The `assets/` folders hold plots and small saved model files produced by the exercises.

## Notes

- Several exercises come from a PyTorch course; the scripts are my worked solutions and experiments.
- Saved weights (`*.pth`) are small demo models, not pretrained checkpoints.
