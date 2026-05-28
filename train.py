import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.change_dataset import ChangeDetectionDataset
from models.siamese_unet import SiameseUNet
from utils.losses import BCEDiceLoss


# =========================
# PATHS
# =========================

train_pre_path = "data/train/pre-event"
train_post_path = "data/train/post-event"
train_mask_path = "data/train/target"


# =========================
# DATASET
# =========================

train_dataset = ChangeDetectionDataset(
    pre_dir=train_pre_path,
    post_dir=train_post_path,
    mask_dir=train_mask_path
)


train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# =========================
# MODEL
# =========================

model = SiameseUNet().to(device)

print("Model Loaded Successfully!\n")


# =========================
# LOSS FUNCTION
# =========================

criterion = BCEDiceLoss()


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# =========================
# TRAINING
# =========================

num_epochs = 2


for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(train_loader)

    for pre_img, post_img, mask in progress_bar:

        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        mask = mask.to(device)

        # Forward
        outputs = model(pre_img, post_img)

        # Loss
        loss = criterion(outputs, mask)

        # Backprop
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_description(
            f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}"
        )

    epoch_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Average Loss: {epoch_loss:.4f}")


# =========================
# SAVE MODEL
# =========================

os.makedirs(
    "outputs/checkpoints",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "outputs/checkpoints/best_model.pth"
)

print("\nModel Saved Successfully!")

print("\nTraining Completed Successfully!")