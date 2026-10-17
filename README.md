# Project 2: Deep Learning Applications

**Name:** Muhammad Luqman Khan | **Course:** Introduction to Applied AI

## Project 2, Week 5: Scene Classification with Transfer Learning

A ResNet-18 pretrained on ImageNet, fine-tuned to recognise 6 kinds of natural and urban scenes: **buildings, forest, glacier, mountain, sea, street**.

### Results at a glance
| | |
|---|---|
| **Dataset** | Intel Image Classification, 6 classes, **11,929 train / 2,105 validation / 3,000 test** |
| **Model** | ResNet-18 pretrained on ImageNet |
| **Feature extraction** (3,078 trainable parameters) | validation accuracy **0.901** |
| **Fine-tuning layer4** (about 8.4M trainable) | validation accuracy **0.931** |
| **Test accuracy** (used once) | **0.935** |
| **Weakest class** | glacier (recall 0.870), confused with mountain (56 of 553 glacier images) |

### What I did
- Treated an image as a tensor and **reproduced a convolution by hand** (`[30, 30, 0]`)
- Built train, validation and test pipelines with **augmentation on training data only**
- Trained a new head (**3,078 parameters**), then **fine-tuned the last block** with two learning rates (1e-4 and 1e-3)
- Kept the best validation epoch, tested **once**, and analysed errors with a confusion matrix and an error gallery

### Most instructive error
A stone cottage in a green valley, labelled *buildings*, was predicted as *mountain* (0.71). The landscape dominates the photo, so the label is arguably ambiguous, and the model's error is understandable.

### Files
- `week-5-transfer-learning.ipynb`: full notebook with outputs and written analysis
- `resnet18_intel.pt`: trained weights (44.8 MB)

## Week 6 Proposal
**Problem:** classify plant leaf photos into healthy or specific disease categories. **Who benefits:** farmers and agricultural extension workers, who can spot crop disease early with only a phone photo. **Dataset:** PlantVillage on Kaggle (link: [add exact Kaggle URL]), license: [add license shown on the dataset page]. **Size:** [N] classes and about [N] images, with several hundred labelled images per class. I will reuse this week's pipeline: ResNet-18, augmentation on training data only, feature extraction, then fine-tuning.
