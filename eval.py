import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix
)

from datasets.change_dataset import ChangeDetectionDataset
from models.siamese_unet import SiameseUNet


# =========================
# PATHS
# =========================

test_pre_path = "data/test/pre-event"
test_post_path = "data/test/post-event"
test_mask_path = "data/test/target"


# =========================
# DATASET
# =========================

test_dataset = ChangeDetectionDataset(
    pre_dir=test_pre_path,
    post_dir=test_post_path,
    mask_dir=test_mask_path
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
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

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!\n")


# =========================
# STORAGE
# =========================

all_preds = []
all_targets = []


# =========================
# EVALUATION
# =========================

with torch.no_grad():

    for pre_img, post_img, mask in test_loader:

        pre_img = pre_img.to(device)
        post_img = post_img.to(device)

        # Forward
        outputs = model(pre_img, post_img)

        # Sigmoid + Threshold
        preds = torch.sigmoid(outputs)

        preds = (preds > 0.5).float()

        # VALID REGION MASKING
        valid_mask = (post_img > 0.05).float()

        preds = preds * valid_mask[:,0:1,:,:]

        # Convert To NumPy
        preds = preds.cpu().numpy()

        mask = mask.cpu().numpy()

        # Force Binary
        preds = (preds > 0).astype(np.uint8)

        mask = (mask > 0).astype(np.uint8)

        # Flatten
        preds = preds.flatten()

        mask = mask.flatten()

        # Store
        all_preds.extend(preds.tolist())

        all_targets.extend(mask.tolist())


# =========================
# FINAL ARRAYS
# =========================

all_preds = np.array(all_preds)

all_targets = np.array(all_targets)


# =========================
# METRICS
# =========================

iou = jaccard_score(
    all_targets,
    all_preds
)

precision = precision_score(
    all_targets,
    all_preds
)

recall = recall_score(
    all_targets,
    all_preds
)

f1 = f1_score(
    all_targets,
    all_preds
)

cm = confusion_matrix(
    all_targets,
    all_preds
)


# =========================
# RESULTS
# =========================

print("\n===== TEST RESULTS =====\n")

print(f"IoU Score       : {iou:.4f}")

print(f"Precision Score : {precision:.4f}")

print(f"Recall Score    : {recall:.4f}")

print(f"F1 Score        : {f1:.4f}")

print("\n===== CONFUSION MATRIX =====\n")

print(cm)