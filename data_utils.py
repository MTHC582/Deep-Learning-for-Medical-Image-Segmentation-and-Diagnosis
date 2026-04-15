# HELPER CLASS MODULE
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

# ----------------------------------------
# ----------------------------------------
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ]
)


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, indx):
        image_name = self.images[indx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(
            self.mask_dir, image_name.replace(".jpg", "_Segmentation.png")
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Im-order to mantain fixed dimensions of image for nn.
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # APPLYING AUGMENTATION
        # This line applies same transformations to both image and mask.
        augmented = transform(image=image, mask=mask)
        # Making multiple copies of the same image and ending up with increased
        # length of dataset, Hence resulting in more training data...
        image = augmented["image"]
        mask = augmented["mask"]

        image = image / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            mask, dtype=torch.float32
        )


# ----------------------------------------
# ----------------------------------------
