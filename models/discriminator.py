"""
Discriminator — PatchGAN Architecture
Instead of asking "is the whole image real or fake?"
PatchGAN asks "is every 70×70 patch real or fake?"
This gives the Generator much sharper, more local feedback,
which is why Pix2Pix outputs look detailed rather than blurry.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Input  : two RGB images concatenated on channel axis → [B, 6, 256, 256]
             (the original image + the generated/real target image)
    Output : probability map  [B, 1, 30, 30]
             each value = probability that the corresponding patch is real
    """

    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, normalise=True, stride=2):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
            if normalise:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # No BatchNorm on first layer
            *block(6,    64,  normalise=False),   # 256→128
            *block(64,  128),                      # 128→64
            *block(128, 256),                      # 64→32
            *block(256, 512, stride=1),            # 32→31  (stride 1 here)
            nn.Conv2d(512, 1, 4, 1, 1),            # 31→30  final patch map
            nn.Sigmoid(),                          # output 0–1 per patch
        )

    def forward(self, source_image, target_image):
        """
        source_image : the original input image   [B, 3, 256, 256]
        target_image : real or generated image    [B, 3, 256, 256]
        """
        x = torch.cat([source_image, target_image], dim=1)  # → [B, 6, H, W]
        return self.model(x)


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# python models/discriminator.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    D = Discriminator()
    total_params = sum(p.numel() for p in D.parameters())
    print(f"Discriminator parameters: {total_params:,}")

    src  = torch.randn(1, 3, 256, 256)
    tgt  = torch.randn(1, 3, 256, 256)
    out  = D(src, tgt)
    print(f"Input  shapes : {src.shape}, {tgt.shape}")
    print(f"Output shape  : {out.shape}")   # should be [1, 1, 30, 30]
    print("✅ Discriminator OK")
