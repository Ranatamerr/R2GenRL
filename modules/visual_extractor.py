import os
import torch
import torch.nn as nn
import timm


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.freeze_vit = getattr(args, 'freeze_vit', True)
        self.partial_freeze = getattr(args, 'partial_freeze', False)

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)

        ckpt_path = os.environ.get(
            'BIOMEDCLIP_CKPT_PATH',
            '/content/drive/MyDrive/Bachelor/hf_cache/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/9f341de24bfb00180f1b847274256e9b65a3a32e/open_clip_pytorch_model.bin'
        )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        visual_weights = {k[len('visual.trunk.'):]: v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        missing, unexpected = self.vit.load_state_dict(visual_weights, strict=False)
        print(f"Visual weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        if self.freeze_vit:
            # Mode 1: fully frozen
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()
        elif self.partial_freeze:
            # Mode 2: freeze all except last 4 transformer blocks + norm
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.vit.blocks[-4:].parameters():
                param.requires_grad = True
            for param in self.vit.norm.parameters():
                param.requires_grad = True
            # Keep frozen blocks in eval mode, unfrozen in train mode
            self.vit.eval()
            for block in self.vit.blocks[-4:]:
                block.train()
            trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.vit.parameters())
            print(f"Partial freeze: {trainable:,} / {total:,} ViT params trainable")
        # else: fully trainable — no action needed

        # Single trainable projection for patch tokens: 768 → d_vf
        self.proj = nn.Linear(768, args.d_vf)

    def train(self, mode=True):
        super(VisualExtractor, self).train(mode)
        if self.freeze_vit:
            # Mode 1: fully frozen — keep entire ViT in eval
            self.vit.eval()
        elif self.partial_freeze:
            # Mode 2: frozen blocks stay in eval, last 4 blocks follow mode
            self.vit.eval()
            for block in self.vit.blocks[-4:]:
                block.train(mode)
            self.vit.norm.train(mode)
        return self

    def forward(self, images):
        if self.freeze_vit:
            with torch.no_grad():
                # (B, 197, 768): CLS token + 196 patch tokens
                feats = self.vit.forward_features(images)
        else:
            feats = self.vit.forward_features(images)

        patch_feats = feats[:, 1:, :]             # (B, 196, 768)
        patch_feats = self.proj(patch_feats)       # (B, 196, d_vf)
        fc_feats = patch_feats.mean(dim=1)         # (B, d_vf)

        return patch_feats, fc_feats
