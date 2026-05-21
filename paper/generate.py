import os
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# SETTINGS
# -----------------------------
START_INDEX = 0      # change to 10 later
MAX_IMAGES = 10

OUTDIR = "/workspace/wrong_predictions"

# -----------------------------
# DATASET
# -----------------------------
class DummyDataset(Dataset):
    def __init__(self, size=500):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand(3, 32, 32)
        y = torch.randint(0, 10, (1,)).item()
        return x, y, idx

os.makedirs(OUTDIR, exist_ok=True)

dataset = DummyDataset()
loader = DataLoader(dataset, batch_size=4)

saved = 0
seen_wrong = 0

print("Generating readable wrong-prediction images...")

for batch_idx, (images, labels, indices) in enumerate(loader):

    preds = torch.randint(0, 10, labels.shape)

    for i in range(len(images)):

        true_class = labels[i].item()
        pred_class = preds[i].item()

        # only wrong predictions
        if pred_class != true_class:

            # skip earlier wrong predictions
            if seen_wrong < START_INDEX:
                seen_wrong += 1
                continue

            absolute_idx = batch_idx * loader.batch_size + i

            img = images[i].permute(1, 2, 0).numpy()

            plt.figure(figsize=(5,5))

            plt.imshow(img)

            plt.title(
                f"Absolute Index: {absolute_idx}\n"
                f"True Class: {true_class}\n"
                f"Predicted Class: {pred_class}",
                fontsize=12
            )

            plt.axis("off")

            out_path = (
                f"{OUTDIR}/wrong_{absolute_idx}.png"
            )

            plt.savefig(
                out_path,
                dpi=100,
                bbox_inches="tight"
            )

            plt.close("all")

            print("Saved:", out_path)

            saved += 1
            seen_wrong += 1

            if saved >= MAX_IMAGES:
                break

    if saved >= MAX_IMAGES:
        break

print(f"Finished. Saved {saved} images.")
