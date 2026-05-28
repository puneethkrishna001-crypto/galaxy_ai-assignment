import os
import cv2
import numpy as np
import tifffile as tiff

import torch
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):

    def __init__(self, pre_dir, post_dir, mask_dir, transform=None):

        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir

        self.transform = transform

        self.files = sorted(os.listdir(pre_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        # Load Images
        pre_img = tiff.imread(
            os.path.join(self.pre_dir, file_name)
        )

        post_img = tiff.imread(
            os.path.join(self.post_dir, file_name)
        )

        mask = tiff.imread(
            os.path.join(self.mask_dir, file_name)
        )

        # Resize Images
        pre_img = cv2.resize(pre_img, (256, 256))

        post_img = cv2.resize(post_img, (256, 256))

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize Images
        pre_img = pre_img.astype(np.float32) / 255.0

        post_img = post_img.astype(np.float32) / 255.0

        # Convert SAR to 3-channel
        post_img = np.stack(
            [post_img] * 3,
            axis=-1
        )

        # Convert Mask
        mask = mask.astype(np.float32)

        # Transformations
        if self.transform:

            augmented = self.transform(
                image=pre_img,
                image_post=post_img,
                mask=mask
            )

            pre_img = augmented["image"]

            post_img = augmented["image_post"]

            mask = augmented["mask"]

        # Convert To Tensor
        pre_img = torch.tensor(pre_img).permute(2, 0, 1)

        post_img = torch.tensor(post_img).permute(2, 0, 1)

        mask = torch.tensor(mask).unsqueeze(0)

        return pre_img, post_img, mask