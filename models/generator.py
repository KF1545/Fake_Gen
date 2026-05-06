"""
Generator — Pix2Pix U-Net Architecture
Takes any input image and transforms it.
The U-Net design preserves spatial detail through skip connections,
which is why it handles precise edits (tattoos, object removal,
background changes) better than a plain encoder-decoder.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """One downsampling step in the encoder."""

    def __init__(self, in_ch, out_ch, normalise=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalise:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """One upsampling step in the decoder."""

    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Pix2Pix Generator (U-Net 256→256).

    Input  : RGB image  [B, 3, 256, 256]
    Output : RGB image  [B, 3, 256, 256]

    The encoder compresses the image into a small bottleneck,
    the decoder rebuilds it with the desired transformation applied.
    Skip connections (torch.cat) pass fine detail from encoder to
    decoder so edges and textures stay sharp.
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        # First block has no BatchNorm (input is raw pixels)
        self.e1 = ConvBlock(3,    64,  normalise=False)  # 256→128
        self.e2 = ConvBlock(64,  128)                    # 128→64
        self.e3 = ConvBlock(128, 256)                    # 64→32
        self.e4 = ConvBlock(256, 512)                    # 32→16
        self.e5 = ConvBlock(512, 512)                    # 16→8
        self.e6 = ConvBlock(512, 512)                    # 8→4
        self.e7 = ConvBlock(512, 512)                    # 4→2

        # ── Bottleneck ───────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),               # 2→1
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────
        # in_ch doubles at each step because of skip connections (cat)
        self.d1 = UpBlock(512,  512, dropout=True)      # 1→2
        self.d2 = UpBlock(1024, 512, dropout=True)      # 2→4
        self.d3 = UpBlock(1024, 512, dropout=True)      # 4→8
        self.d4 = UpBlock(1024, 512)                    # 8→16
        self.d5 = UpBlock(1024, 256)                    # 16→32
        self.d6 = UpBlock(512,  128)                    # 32→64
        self.d7 = UpBlock(256,   64)                    # 64→128

        # Final layer: back to 3-channel RGB
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),        # 128→256
            nn.Tanh(),   # output range [-1, 1]
        )

    def forward(self, x):
        # Encode — save each output for skip connections
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decode — concatenate matching encoder features at each step
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))

        return self.final(torch.cat([d7, e1], dim=1))


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# python models/generator.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = Generator()
    total_params = sum(p.numel() for p in G.parameters())
    print(f"Generator parameters: {total_params:,}")

    dummy = torch.randn(1, 3, 256, 256)
    out = G(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")    # should be [1, 3, 256, 256]
    print("✅ Generator OK")
