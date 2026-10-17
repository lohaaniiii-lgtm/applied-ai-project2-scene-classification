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

## Week 6 Proposal: Household Waste Classification

**Problem:** classify a photo of a waste item into one of 12 household garbage categories (paper, cardboard, biological, metal, plastic, green glass, brown glass, white glass, clothes, shoes, batteries and trash).
**Who benefits:** municipal waste teams, recycling collectors and housing societies or universities in Pakistani cities, where mixed waste makes recycling difficult. Sorting waste correctly at the source would make recycling easier and cut landfill waste.
**Dataset:** [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Mohamed on Kaggle, license: Open Database License (ODbL).
**Size:** 12 classes and 15,150 images, an average of about 1,260 per class, well above the few hundred per class the lab requires.
**Method:** the same pipeline as this week: ResNet-18, augmentation on the training data only, feature extraction, then fine-tuning. The images were mostly collected by web scraping, so I will also test the model on photos of waste around me to see how well it handles real backgrounds.
