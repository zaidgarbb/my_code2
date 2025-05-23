# train_segmentation.py content
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import LabeledDataset
from split_data import split_dataset

# ✅ مسار البيانات في EC2
BASE_PATH = "/home/ubuntu/semi_supervised_fixmatch_all_scripts/BraTS_FixMatch/BraTS_FixMatch"
IMAGE_DIR = os.path.join(BASE_PATH, 'labeled/images')
MASK_DIR = os.path.join(BASE_PATH, 'labeled/masks')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'best_model.pth')

# ⚙️ الإعدادات
EPOCHS = 20
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔀 تقسيم البيانات إلى train / val
train_data, val_data, _ = split_dataset(IMAGE_DIR, MASK_DIR)

train_dataset = LabeledDataset(
    [x[0] for x in train_data],
    [x[1] for x in train_data],
    augment=True
)

val_dataset = LabeledDataset(
    [x[0] for x in val_data],
    [x[1] for x in val_data],
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 🧠 تعريف النموذج
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

# 🎯 خسارة ومحسن
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')

# 🔁 التدريب
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for imgs, masks in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE).unsqueeze(1)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    # 🧪 التحقق على val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            val_loss += criterion(outputs, masks).item()

    val_loss /= len(val_loader)
    print(f"📉 Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Model saved with best val loss: {best_val_loss:.4f}")
