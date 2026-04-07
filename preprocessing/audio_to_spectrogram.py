"""
Convert audio files into mel spectrogram images.

This script uses parsed protocol files to generate spectrograms
for specific time windows aligned with video segments.
"""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_ROOT = "data"
AUDIO_DIR = os.path.join(DATA_ROOT, "audio_files")
SPECTROGRAM_DIR = os.path.join(DATA_ROOT, "spectrograms")
FACE_DIR = os.path.join(DATA_ROOT, "face_frames")
PARSED_PROTOCOL_FILE = os.path.join(DATA_ROOT, "parsed_protocols", "parsed_train.csv")


def generate_segment_spectrograms(
    audio_dir: str = AUDIO_DIR,
    output_dir: str = SPECTROGRAM_DIR,
    face_dir: str = FACE_DIR,
    parsed_protocol_file: str = PARSED_PROTOCOL_FILE,
    sample_rate: int = 16000,
) -> None:
    """
    Generate mel spectrogram images for parsed clip segments.
    """
    os.makedirs(output_dir, exist_ok=True)

    parsed_files = pd.read_csv(parsed_protocol_file, header=None)

    for i in range(parsed_files.shape[0]):
        sample_path, _, start_frame, end_frame = parsed_files.iloc[i]

        total_frames = len(os.listdir(os.path.join(face_dir, sample_path)))
        base_name = sample_path.split("/")[0]

        start_time = float(start_frame / total_frames)
        end_time = float(end_frame / total_frames)

        audio_file_path = os.path.join(audio_dir, base_name + ".wav")

        if not os.path.exists(audio_file_path):
            print(f"Missing audio file: {audio_file_path}")
            continue

        try:
            y, sr = librosa.load(audio_file_path, sr=sample_rate)
            y = y[int(start_time * sample_rate): int(end_time * sample_rate)]

            mel_spect = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=128,
            )

            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            save_name = f"{base_name}_{start_frame}_{end_frame}.png"
            save_path = os.path.join(output_dir, save_name)

            plt.figure(figsize=(6.4, 4.8))
            librosa.display.specshow(mel_spect, fmax=sr)
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            print(f"Saved: {save_name}")

        except Exception as e:
            print(f"Failed on {audio_file_path}: {e}")


if __name__ == "__main__":
    generate_segment_spectrograms()
