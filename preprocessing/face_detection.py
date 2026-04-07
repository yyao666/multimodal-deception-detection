"""
Extract and align face frames from video clips.

This script reads recorded videos, detects/aligs faces frame-by-frame,
and saves aligned face images for downstream visual modeling.
"""

import os
import cv2
from tools import FaceAlignmentTools  

# from preprocessing.face_alignment_tools import FaceAlignmentTools


DATA_ROOT = "data"
RECORDINGS_DIR = os.path.join(DATA_ROOT, "recordings")
SPECTROGRAM_DIR = os.path.join(DATA_ROOT, "spectrograms")
FACE_DIR = os.path.join(DATA_ROOT, "face_frames")


def extract_face_frames(
    recordings_dir: str = RECORDINGS_DIR,
    spectrogram_dir: str = SPECTROGRAM_DIR,
    output_dir: str = FACE_DIR,
) -> None:
    """
    Extract aligned face frames from all video clips.

    Only videos with corresponding spectrogram files are processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    spec_files = set(os.listdir(spectrogram_dir))
    already_processed = set(os.listdir(output_dir))

    tool = FaceAlignmentTools()

    total_vids = len(spec_files)
    processed_count = len(already_processed)

    for subject in os.listdir(recordings_dir):
        subject_video_dir = os.path.join(recordings_dir, subject, "video_clips")

        if not os.path.isdir(subject_video_dir):
            continue

        for file_name in os.listdir(subject_video_dir):
            video_path = os.path.join(subject_video_dir, file_name)
            clip_name = os.path.splitext(file_name)[0]
            sample_name = f"{subject}_{clip_name}"

            if f"{sample_name}.png" not in spec_files:
                continue

            if sample_name in already_processed:
                print(f"Skipping existing sample: {sample_name}")
                continue

            save_path = os.path.join(output_dir, sample_name)
            os.makedirs(save_path, exist_ok=True)

            print(
                f"Processing: {sample_name} | "
                f"Progress: {round(processed_count / max(total_vids, 1), 2)}"
            )
            processed_count += 1

            cap = cv2.VideoCapture(video_path)
            num_frames = 0

            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aligned_img = tool.align(frame)

                if aligned_img is None:
                    continue

                aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
                num_frames += 1

                frame_id = f"{num_frames:04d}"
                save_image = os.path.join(save_path, f"frame_{frame_id}.jpg")
                cv2.imwrite(save_image, aligned_img)

            cap.release()


if __name__ == "__main__":
    extract_face_frames()
