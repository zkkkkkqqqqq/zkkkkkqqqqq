import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("FAIL: PyTorch cannot access CUDA.")

print("GPU:", torch.cuda.get_device_name(0))

values = torch.tensor([1, 2, 3], device="cuda")
result = values.sum().item()

if result != 6:
    raise SystemExit(f"FAIL: Expected 6, got {result}.")

print("PASS: CUDA calculation returned 6.")
