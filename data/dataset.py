"""
Dataset utilities for multimodal deception detection.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ParsedFaceSpecDataset(Dataset):
    """
    Dataset for multimodal deception detection using:
    - audio spectrogram tensors
    - extracted face-frame sequences
    - parsed protocol annotations
    """

    def __init__(
        self,
        annotations_file: str,
        spec_dir: str,
        img_dir: str,
        time_window: float,
        fps: int,
    ):
        self.annotations = pd.read_csv(annotations_file, header=None)
        self.spec_dir = spec_dir
        self.img_dir = img_dir
        self.time_window = time_window
        self.fps = fps
        self.num_target_frames = int(time_window * fps)

        self.visual_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample_path = self.annotations.iloc[idx, 0]
        label_token = self.annotations.iloc[idx, 1]
        start_index = self.annotations.iloc[idx, 2]
        end_index = self.annotations.iloc[idx, 3]

        frame_dir = os.path.join(self.img_dir, sample_path)
        frame_names = sorted(os.listdir(frame_dir))
        total_frames = len(frame_names)

        window_frames = frame_names[start_index:end_index]

        if len(window_frames) == 0:
            raise ValueError(f"No frames found for sample segment: {sample_path}")

        target_indices = np.linspace(
            start=0,
            stop=len(window_frames) - 1,
            num=self.num_target_frames,
            dtype=np.int32,
        )

        face_frames = []
        for i in target_indices:
            img = np.asarray(
                Image.open(os.path.join(frame_dir, window_frames[i])),
                dtype=np.float32,
            ) / 255.0
            face_frames.append(self.visual_transforms(img))

        face_frames = torch.stack(face_frames, 0).permute(1, 0, 2, 3)

        raw_spec = torch.load(os.path.join(self.spec_dir, sample_path + ".pth"))
        spec_length = raw_spec.shape[-1]
        spec_start = round(spec_length * (start_index / total_frames))
        spec_end = round(spec_length * (end_index / total_frames))
        window_spec = raw_spec[:, spec_start:spec_end]

        if label_token == "T":
            label = 0
        elif label_token in ["F", "L"]:
            label = 1
        else:
            raise ValueError(f"Unknown label token: {label_token}")

        label = torch.tensor(label, dtype=torch.long)

        clip_parts = sample_path.split("/")
        clip_name = clip_parts[1] if len(clip_parts) > 1 else clip_parts[0]

        return clip_name, window_spec, face_frames, label


def pad_spectrogram_sequence(batch):
    """
    Pad variable-length spectrogram tensors to the same temporal length.
    """
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.0
    )
    return batch.permute(0, 2, 1)


def multimodal_collate_fn(batch):
    """
    Custom collate function for multimodal batching.
    """
    clip_names, spec_tensors, face_tensors, labels = [], [], [], []

    for clip_name, spec, face_frames, label in batch:
        clip_names.append(clip_name)
        spec_tensors.append(spec)
        face_tensors.append(face_frames)
        labels.append(label)

    spec_tensors = pad_spectrogram_sequence(spec_tensors)
    face_tensors = torch.stack(face_tensors)
    labels = torch.stack(labels)

    return clip_names, spec_tensors, face_tensors, labels
