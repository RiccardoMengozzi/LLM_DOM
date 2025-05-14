#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        # Properly registered submodules
        self.seq = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.seq(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(in_features=20, hidden=10, out_features=2).to(device)

    # Verify parameters are found
    params = list(model.parameters())
    print(f"Found {len(params)} parameter tensors.")  # should be non-zero

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Dummy training loop
    x = torch.randn(8, 20).to(device)
    y = torch.randint(0, 2, (8,)).to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print("Training step completed without errors.")

if __name__ == "__main__":
    main()
