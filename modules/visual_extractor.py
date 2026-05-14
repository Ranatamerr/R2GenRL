import torch
import torch.nn as nn
import open_clip


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.vit = model.visual
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        # Single trainable projection for patch tokens: 768 → d_vf
        self.proj = nn.Linear(768, args.d_vf)

    def train(self, mode=True):
        super(VisualExtractor, self).train(mode)
        # Keep ViT in eval mode regardless of parent training mode
        self.vit.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            # (B, 197, 768): CLS token + 196 patch tokens
            feats = self.vit.trunk.forward_features(images)

        patch_feats = feats[:, 1:, :]             # (B, 196, 768)
        patch_feats = self.proj(patch_feats)       # (B, 196, d_vf)
        fc_feats = patch_feats.mean(dim=1)         # (B, d_vf)

        return patch_feats, fc_feats
