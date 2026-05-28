import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

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
    shuffle=True
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

print("Model Loaded Successfully!")


# =========================
# OUTPUT DIRECTORY
# =========================

os.makedirs(
    "outputs/predictions",
    exist_ok=True
)


# =========================
# INFERENCE
# =========================

with torch.no_grad():

    for idx, (pre_img, post_img, mask) in enumerate(test_loader):

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
        pre_np = pre_img[0].cpu().permute(1,2,0).numpy()

        post_np = post_img[0].cpu().permute(1,2,0).numpy()

        mask_np = mask[0][0].cpu().numpy()

        pred_np = preds[0][0].cpu().numpy()


        # Plot
        fig, ax = plt.subplots(1,4, figsize=(20,5))

        ax[0].imshow(pre_np)
        ax[0].set_title("Pre-event")

        ax[1].imshow(post_np[:,:,0], cmap='gray')
        ax[1].set_title("Post-event SAR")

        ax[2].imshow(mask_np, cmap='gray')
        ax[2].set_title("Ground Truth")

        ax[3].imshow(pred_np, cmap='gray')
        ax[3].set_title("Prediction")


        for a in ax:
            a.axis("off")


        plt.tight_layout()

        # Save Prediction
        plt.savefig(
            f"outputs/predictions/prediction_{idx}.png"
        )

        plt.close()

        print(f"Saved Prediction {idx}")

        # Save First 10 Only
        if idx == 9:
            break


print("\nInference Completed Successfully!")