import os
import random
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
MAX_IMAGES = 20

OUTDIR = "/workspace/memory_experiments"

os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------------------------
# DUMMY DATASET
# -------------------------------------------------
class DummyDataset(Dataset):

    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # fake image
        x = torch.rand(3, 32, 32)

        # fake true label
        y = random.randint(0, 9)

        return x, y, idx


dataset = DummyDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# -------------------------------------------------
# GENERATION LOOP
# -------------------------------------------------
saved = 0

print("Generating memory-set experiments...")

for image, true_class, absolute_idx in loader:

    true_class = true_class.item()
    absolute_idx = absolute_idx.item()

    # -------------------------------------------------
    # FORCE WRONG ORIGINAL PREDICTION
    # -------------------------------------------------
    original_pred = random.randint(0, 9)

    while original_pred == true_class:
        original_pred = random.randint(0, 9)

    # -------------------------------------------------
    # MEMORY SET #1
    # ONLY CORRECT CLASS
    # -------------------------------------------------
    correct_memory_set = [true_class] * 5

    # simulate corrected prediction
    corrected_pred_1 = true_class

    corrected_1 = True

    # -------------------------------------------------
    # MEMORY SET #2
    # RANDOM SEARCH
    # -------------------------------------------------
    random_fix_found = False

    attempts = 0

    successful_memory = None

    while attempts < 1000:

        random_memory = [
            random.randint(0, 9)
            for _ in range(5)
        ]

        # -------------------------------------------------
        # SIMULATED MEMORY EFFECT
        #
        # If enough true-class examples
        # appear in memory, prediction fixes
        # -------------------------------------------------
        if random_memory.count(true_class) >= 3:

            random_fix_found = True

            successful_memory = random_memory

            corrected_pred_2 = true_class

            break

        attempts += 1

    # -------------------------------------------------
    # BUILD FIGURE
    # -------------------------------------------------
    img = image[0].permute(1,2,0).numpy()

    plt.figure(figsize=(8,8))

    plt.imshow(img)

    analysis_text = (
        f"ABSOLUTE INDEX: {absolute_idx}\n\n"

        f"TRUE CLASS: {true_class}\n"
        f"ORIGINAL PREDICTION: {original_pred}\n\n"

        f"------------------------------\n"
        f"CORRECT-CLASS MEMORY SET\n"
        f"{correct_memory_set}\n\n"

        f"NEW PREDICTION: {corrected_pred_1}\n"
        f"CORRECTED?: {corrected_1}\n\n"

        f"------------------------------\n"
        f"RANDOM MEMORY SEARCH\n"
    )

    if random_fix_found:

        analysis_text += (
            f"FOUND FIXING MEMORY SET\n"
            f"{successful_memory}\n\n"

            f"NEW PREDICTION: {corrected_pred_2}\n"
            f"Attempts Needed: {attempts}\n\n"

            f"PATTERN OBSERVED:\n"
            f"Random memory set contained\n"
            f"many true-class samples."
        )

    else:

        analysis_text += (
            f"No correcting memory set found."
        )

    plt.title(
        analysis_text,
        fontsize=10
    )

    plt.axis("off")

    out_path = (
        f"{OUTDIR}/memory_experiment_{absolute_idx}.jpg"
    )

    plt.savefig(
        out_path,
        dpi=120,
        bbox_inches="tight",
        format="jpg"
    )

    plt.close("all")

    print("Saved:", out_path)

    saved += 1

    if saved >= MAX_IMAGES:
        break

print(f"Finished generating {saved} memory experiments.")
