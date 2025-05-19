# train_emojis.py
import torch, torchvision
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_ROOT   = Path("emoji_dataset")
IMG_SIZE    = 72           # your images are already 72×72
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────
# 1️⃣  Transforms
# ─────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomRotation(12),                # ±20° (faces look OK)
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.9, 1.25)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
])

val_tf   = transforms.ToTensor()

# ─────────────────────────────
# 2️⃣  Datasets & loaders
# ─────────────────────────────
train_ds = ImageFolder(DATA_ROOT / "train", transform=train_tf)
val_ds   = ImageFolder(DATA_ROOT / "val",   transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = train_ds.classes    # e.g. ['U+1F600', 'U+1F602', …]

# ─────────────────────────────
# 3️⃣  A tiny CNN
# ─────────────────────────────
class TinyEmojiNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),  # 72×72
            nn.MaxPool2d(2),                            # 36×36
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                            # 18×18
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                            # 9×9
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                     # 1×1
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

model     = TinyEmojiNet(len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────
# 4️⃣  Training loop
# ─────────────────────────────
def run_epoch(dl, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.inference_mode() if not train else torch.enable_grad():
        for X, y in dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            preds     = logits.argmax(1)
            correct  += (preds == y).sum().item()
            total    += y.size(0)
    return loss_sum / total, correct / total

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_dl, train=True)
    val_loss,   val_acc   = run_epoch(val_dl,   train=False)
    print(f"[{epoch:02}/{EPOCHS}]  "
          f"train {train_loss:.3f}/{train_acc:.2%}  |  "
          f"val {val_loss:.3f}/{val_acc:.2%}")

# ─────────────────────────────
# 5️⃣  Save the model
# ─────────────────────────────
torch.save({"model": model.state_dict(),
            "classes": class_names},
           "emoji_cnn.pt")
print("✅ Model saved to emoji_cnn.pt")
