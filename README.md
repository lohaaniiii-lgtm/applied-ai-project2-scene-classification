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

## Week 6 Proposal

## Week 6 Proposal

**Problem:** classify a photo of a waste item into one of 12 categories (paper, cardboard, biological waste, metal, plastic, green glass, brown glass, white glass, clothing, shoes, batteries, general trash).
**Who benefits:** municipal waste teams, recycling collectors and universities or housing societies in Pakistani cities, where mixed waste makes recycling hard. A model like this could help sort waste correctly at the source and support recycling.
**Dataset:** [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) on Kaggle (license: see the dataset page).
**Size:** 12 classes and about 15,000 labelled images (15,150), far more than the few hundred per class the lab requires.
**Method:** the same pipeline as this week: ResNet-18, augmentation on the training data only, feature extraction, then fine-tuning. I will also test the model on photos of waste items around me, such as bottles, cans, paper and batteries, to see how it copes with real backgrounds.
