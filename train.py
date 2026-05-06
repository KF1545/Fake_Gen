"""
train.py — Runs on Google Colab (GPU)
This file never runs on your laptop.
It trains the Pix2Pix GAN on CelebA, saves checkpoints to
Google Drive every few epochs so progress is never lost.

How to run in Colab:
    !python /content/deepfake-app/train.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ── Make sure our models folder is importable ────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.generator     import Generator
from models.discriminator import Discriminator

# ══════════════════════════════════════════════════════════════
# CONFIGURATION  — change these if needed
# ══════════════════════════════════════════════════════════════
CFG = {
    "data_dir":       "/content/drive/MyDrive/deepfake-app/data/celeba/img_align_celeba",
    "checkpoint_dir": "/content/drive/MyDrive/deepfake-app/checkpoints",
    "results_dir":    "/content/drive/MyDrive/deepfake-app/results",
    "image_size":     256,
    "batch_size":     16,
    "epochs":         50,
    "lr":             0.0002,
    "lambda_l1":      100,    # weight of L1 loss vs adversarial loss
    "save_every":     5,      # save checkpoint every N epochs
    "sample_every":   1,      # save sample images every N epochs
    "max_images":     10000,  # cap dataset size to keep Colab fast
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
os.makedirs(CFG["results_dir"],    exist_ok=True)


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class CelebADataset(Dataset):
    """
    Loads CelebA face images.
    Each image is returned twice — as the 'source' and 'target'.
    The Generator learns to reconstruct/transform images.
    During inference, the source is the user's uploaded image.
    """

    def __init__(self, folder, max_images=10000, size=256):
        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:max_images]

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),   # data augmentation
            transforms.ToTensor(),
            # Normalise to [-1, 1] — matches Generator Tanh output
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),
        ])

        print(f"Dataset: {len(self.files)} images found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        tensor = self.transform(img)
        # Return (source, target) — same image used for both during training
        # because we are teaching the GAN to understand image structure
        return tensor, tensor


# ══════════════════════════════════════════════════════════════
# WEIGHTS INITIALISATION
# ══════════════════════════════════════════════════════════════
def init_weights(model):
    """Initialise Conv and BatchNorm layers — standard for GANs."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def train():
    # ── Data ─────────────────────────────────────────────────
    dataset = CelebADataset(
        CFG["data_dir"],
        max_images=CFG["max_images"],
        size=CFG["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ── Models ───────────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    init_weights(G)
    init_weights(D)

    # ── Optimisers ───────────────────────────────────────────
    g_opt = torch.optim.Adam(G.parameters(),
                              lr=CFG["lr"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(),
                              lr=CFG["lr"], betas=(0.5, 0.999))

    # ── Loss functions ───────────────────────────────────────
    bce_loss = nn.BCELoss()
    l1_loss  = nn.L1Loss()

    # ── Resume from checkpoint if one exists ─────────────────
    start_epoch    = 0
    checkpoint_path = os.path.join(CFG["checkpoint_dir"], "latest.pth")

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=DEVICE)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        g_opt.load_state_dict(ck["g_opt"])
        d_opt.load_state_dict(ck["d_opt"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")
    else:
        print("No checkpoint found — starting fresh")

    # ── Training loop ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Training started")
    print("═" * 60)

    for epoch in range(start_epoch, CFG["epochs"]):
        G.train()
        D.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{CFG['epochs']}")

        for source, target in loop:
            source = source.to(DEVICE)
            target = target.to(DEVICE)
            B      = source.size(0)

            # ── Train Discriminator ───────────────────────────
            fake  = G(source)

            # Labels for PatchGAN output  [B, 1, 30, 30]
            real_label = torch.ones (B, 1, 30, 30).to(DEVICE)
            fake_label = torch.zeros(B, 1, 30, 30).to(DEVICE)

            d_real_loss = bce_loss(D(source, target), real_label)
            d_fake_loss = bce_loss(D(source, fake.detach()), fake_label)
            d_loss      = (d_real_loss + d_fake_loss) * 0.5

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ── Train Generator ───────────────────────────────
            # Adversarial: fool the discriminator
            g_adv  = bce_loss(D(source, fake), real_label)
            # Pixel-level: stay close to the original structure
            g_l1   = l1_loss(fake, target) * CFG["lambda_l1"]
            g_loss = g_adv + g_l1

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            loop.set_postfix(D=f"{d_loss.item():.4f}",
                              G=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / len(loader)
        avg_g = epoch_g_loss / len(loader)
        print(f"\nEpoch {epoch} complete | "
              f"Avg D loss: {avg_d:.4f} | Avg G loss: {avg_g:.4f}")

        # ── Save sample images to Drive ───────────────────────
        if epoch % CFG["sample_every"] == 0:
            G.eval()
            with torch.no_grad():
                sample = G(source[:4])
            # Denormalise from [-1,1] back to [0,1] for saving
            comparison = torch.cat([source[:4], sample], dim=0)
            save_path  = os.path.join(
                CFG["results_dir"], f"epoch_{epoch:03d}.png")
            save_image(comparison * 0.5 + 0.5, save_path, nrow=4)
            print(f"Sample saved → {save_path}")

        # ── Save checkpoint to Drive ──────────────────────────
        if epoch % CFG["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "G":     G.state_dict(),
                "D":     D.state_dict(),
                "g_opt": g_opt.state_dict(),
                "d_opt": d_opt.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved → {checkpoint_path}")

    # ── Save final model ──────────────────────────────────────
    final_path = os.path.join(CFG["checkpoint_dir"], "generator_final.pth")
    torch.save(G.state_dict(), final_path)
    print(f"\n{'═'*60}")
    print(f"Training complete!")
    print(f"Final model saved → {final_path}")
    print(f"Download this file to your laptop's checkpoints/ folder.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    train()
