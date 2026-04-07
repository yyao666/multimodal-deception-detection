"""
Multimodal ResNet-50 Deception Detection Model

This module defines a multimodal architecture combining:
- Audio spectrogram features (ResNet-50)
- Visual face-frame features (Slow R50)

Fusion Strategy: (feature-level)
Feature concatenation followed by classification layer
"""

import torch
import torch.nn as nn
import torchvision


class MultimodalR50(nn.Module):
    """
    Multimodal deception detection model

    Architecture:
    - Audio Tower  : ResNet-50 (spectrogram input)
    - Visual Tower : Slow R50 (video face frames)
    - Fusion       : Concatenation
    - Classifier   : Binary classifier
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()

        # -----------------------------
        # Audio Backbone
        # -----------------------------
        audio_backbone = torchvision.models.resnet50(
            weights="ResNet50_Weights.IMAGENET1K_V1"
        )

        # Modify input channel for spectrogram (1 channel)
        audio_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        audio_feat_dim = audio_backbone.fc.in_features
        audio_backbone.fc = nn.Identity()

        self.audio_backbone = nn.Sequential(
            audio_backbone,
            nn.Flatten(),
        )

        # -----------------------------
        # Visual Backbone
        # -----------------------------
        visual_backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slow_r50",
            pretrained=True,
        )

        visual_backbone.blocks[5].proj = nn.Identity()

        self.visual_backbone = nn.Sequential(
            visual_backbone,
            nn.Flatten(),
        )

        visual_feat_dim = 2048

        # -----------------------------
        # Classifier
        # -----------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                audio_feat_dim + visual_feat_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        face_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            spectrogram : audio spectrogram tensor
            face_frames : video face frame tensor

        Returns:
            logits
        """

        spectrogram = spectrogram.float()

        audio_features = self.audio_backbone(
            spectrogram.unsqueeze(1)
        )

        visual_features = self.visual_backbone(
            face_frames
        )

        fused = torch.cat(
            (audio_features, visual_features),
            dim=-1
        )

        return self.classifier(fused)
