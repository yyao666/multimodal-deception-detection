"""
Extract audio tracks from recorded video clips.

This script traverses the video dataset and saves one WAV file
for each video clip. It is intended as the first preprocessing step
for multimodal deception detection.
"""

import os
from moviepy.editor import VideoFileClip


DATA_ROOT = "data"
RECORDINGS_DIR = os.path.join(DATA_ROOT, "recordings")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio_files")


def extract_audio_from_videos(
    recordings_dir: str = RECORDINGS_DIR,
    output_dir: str = AUDIO_DIR,
) -> None:
    """
    Extract audio from all video clips and save as WAV files.

    Expected directory structure:
        data/recordings/<subject>/video_clips/*.mp4
    """
    os.makedirs(output_dir, exist_ok=True)

    completed = set(os.listdir(output_dir))

    for subject in os.listdir(recordings_dir):
        subject_video_dir = os.path.join(recordings_dir, subject, "video_clips")

        if not os.path.isdir(subject_video_dir):
            continue

        for file_name in os.listdir(subject_video_dir):
            video_path = os.path.join(subject_video_dir, file_name)
            clip_name = os.path.splitext(file_name)[0]
            output_name = f"{subject}_{clip_name}.wav"

            if output_name in completed:
                print(f"Skipping existing file: {output_name}")
                continue

            print(f"Processing: {video_path}")

            try:
                video = VideoFileClip(video_path)
                audio = video.audio

                if audio is None:
                    print(f"No audio found in: {video_path}")
                    continue

                output_path = os.path.join(output_dir, output_name)
                audio.write_audiofile(output_path)

                video.close()
                audio.close()

            except Exception as e:
                print(f"Failed on {video_path}: {e}")


if __name__ == "__main__":
    extract_audio_from_videos()
