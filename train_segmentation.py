import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LabeledDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm

# إعدادات
DATA_DIR = "./data"
BATCH_SIZE = 8
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# augmentations
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# تحميل البيانات
# إعداد المسارات الكاملة للصور والماسكات
def get_file_paths(img_dir, mask_dir):
    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if not os.path.isdir(os.path.join(img_dir, f))]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if not os.path.isdir(os.path.join(mask_dir, f))]
    return image_paths, mask_paths

train_img_dir = os.path.join(DATA_DIR, "train/images")
train_mask_dir = os.path.join(DATA_DIR, "train/masks")
val_img_dir = os.path.join(DATA_DIR, "val/images")
val_mask_dir = os.path.join(DATA_DIR, "val/masks")

train_image_paths, train_mask_paths = get_file_paths(train_img_dir, train_mask_dir)
val_image_paths, val_mask_paths = get_file_paths(val_img_dir, val_mask_dir)

train_dataset = LabeledDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = LabeledDataset(val_image_paths, val_mask_paths, transform=transform)



# النموذج
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# دالة لحساب الدقة
def calculate_accuracy(preds, masks):
    preds = (preds > 0.5).float()
    correct = (preds == masks).float()
    return correct.sum() / correct.numel()

# التدريب
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_acc = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        acc = calculate_accuracy(torch.sigmoid(outputs), masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    # التحقق
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            val_acc += calculate_accuracy(torch.sigmoid(outputs), masks).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} — Loss: {avg_loss:.4f} — Acc: {avg_acc*100:.2f}% "
          f"— Val Loss: {avg_val_loss:.4f} — Val Acc: {avg_val_acc*100:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Model saved with best val loss: {best_val_loss:.4f}")
