"""
Protocol parsing utilities for multimodal deception detection.

This module converts original fold-level protocol CSV files into
segment-level samples by splitting long clips into shorter windows.
"""

import os
import numpy as np
import pandas as pd


def parse_protocol_csv(
    annotations_file: str,
    output_txt: str,
    output_csv: str,
    img_dir: str,
    time_window: float,
    mode: str = None,
    fps: float = 5.0,
) -> None:
    """
    Parse a protocol CSV into segment-level samples.

    Each long clip is split into multiple shorter segments whose
    durations are approximately equal to `time_window`.

    Args:
        annotations_file: Path to the original fold CSV file.
        output_txt: Path to intermediate parsed txt file.
        output_csv: Path to final parsed csv file.
        img_dir: Directory containing extracted face-frame folders.
        time_window: Desired segment duration in seconds.
        mode: None, "monologue", or "interrogation".
        fps: Frame rate used when extracted face frames were saved.
    """
    annotations = pd.read_csv(annotations_file)

    with open(output_txt, "w") as f:
        for i in range(annotations.shape[0]):
            sample_path = annotations.iloc[i, 0]
            label = annotations.iloc[i, 5]
            conversation_mode = annotations.iloc[i, 4]

            if mode == "monologue" and conversation_mode == "interrogation":
                continue

            if mode == "interrogation" and conversation_mode in ["mono", "monologue"]:
                continue

            frame_dir = os.path.join(img_dir, sample_path)
            num_frames = len(os.listdir(frame_dir))
            duration = round(num_frames / fps, 2)

            if duration <= time_window:
                f.write(f"{sample_path},{label},0,{num_frames}\n")
            else:
                num_splits = round(duration / time_window)
                split_indices = np.array_split(np.arange(num_frames), num_splits)

                for split in split_indices:
                    f.write(f"{sample_path},{label},{split[0]},{split[-1]}\n")

    parsed = pd.read_csv(output_txt, header=None)
    parsed.to_csv(output_csv, index=False, header=False) 
