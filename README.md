# multimodal-deception-detection


## 🎯 Research Objective

The goal of this research is to develop a **robust and fair multimodal deception detection system** that generalizes across:

* Different speakers
* Different ethnic groups
* Different languages

This work focuses on cross-domain generalization and representation robustness in multimodal learning.



## 🧠 Overview

The baseline implementation includes:

* Audio feature learning from spectrogram inputs using **ResNet-50**
* Visual feature learning from face-frame sequences using **Slow R50**
* Multimodal feature fusion via **feature concatenation**
* Segment-level training for long video clips
* Clip-level evaluation via segment aggregation
* Fold-based protocol evaluation


## Model Architecture

The baseline model consists of two modality-specific backbones.

🔊 Audio Tower
* ResNet-50 adapted for single-channel spectrogram input
* Learns acoustic deception-related representations

🎥 Visual Tower
* Pretrained Slow R50 (3D CNN)
* Processes face-frame sequences extracted from videos

🔗 Fusion Strategy
* Audio and visual features are concatenated
* Binary classifier outputs deception prediction


## 📁 Repository Structure


```
multimodal-deception-detection/
│
├── models/
│   └── multimodal_r50.py
│
├── preprocessing/
│   ├── video_to_audio.py
│   ├── audio_to_spectrogram.py
│   ├── face_detection.py
│   ├── face_alignment_tools.py
│   └── helper.py
│
├── data/
│   ├── dataset.py
│   └── parsing.py
│
├── train.py
├── evaluate.py
├── config.yaml
├── requirements.txt
└── README.md
```

## 📊 Dataset

⚠️ **Note**: The dataset is **not publicly released** due to privacy and licensing constraints.

This repository provides:

* Data preprocessing scripts
* Dataset loading utilities
* Training pipeline

To run the code, the following resources are expected:

* Protocol CSV files
* Extracted face-frame folders
* Precomputed spectrogram tensors (.pth)

## 📂 Expected Dataset Structure


```
dataset/
├── protocols/
│   ├── train1.csv
│   ├── test1.csv
│   ├── train2.csv
│   └── test2.csv
│
├── face_frames/
│   └── <sample_path>/
│       ├── frame_0001.jpg
│       ├── frame_0002.jpg
│       └── ...
│
└── spectrograms/
    └── <sample_path>.pth
```


## 📚 Public Datasets Used

* Bag of Lies
* Real-Life Trial Dataset

Additionally, this project includes a **self-collected multimodal dataset** spanning:

* Multiple ethnic groups
* Multiple languages
* Diverse speaking styles

Due to privacy considerations, this dataset is not publicly released.





## 📝 Notes

This public version is intended as:


*   A clean research codebase
*   A reproducible baseline
* A modular research framework

Some preprocessing details may need to be adapted depending on local dataset organization.




## 📄 License

MIT License (see LICENSE file)
