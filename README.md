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

---

## Project 2, Week 6: Recycling Waste Sorting (12-Class Garbage Classification)

A ResNet-18 fine-tuned to sort a photo of a waste item into one of 12 household garbage classes: **battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass**.

**Who benefits:** municipal waste teams, recycling collectors and housing societies or universities in Pakistani cities, where mixed waste makes recycling difficult. **Why it matters:** a wrongly sorted item contaminates a whole recycling stream (for example glass or a battery in the wrong bin).

### Datasheet
| | |
|---|---|
| **Dataset** | [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Mohamed, Kaggle, license ODbL |
| **Size** | 15,515 images, 12 classes |
| **Source and labels** | images mostly collected by web scraping; [add who labelled it, from the dataset page] |
| **Imbalance** | ratio 8.8 (clothes 5,325 vs brown-glass 607) |
| **Audit** | 37 exact duplicate files, 3 sets with conflicting labels, 39 near-duplicate pairs; duplicates handled |
| **Split** | no test folder, so I made a grouped train/val/test split by near-duplicate cluster; 0 shared groups; test set 2,158 images |

### Model card
| | |
|---|---|
| **Model** | ResNet-18 pretrained on ImageNet, two-phase training (new head first, then fine-tuning) |
| **Imbalance handling** | class-weighted loss, weights N / (K × n_c) |
| **Test results (used once)** | macro-F1 **0.928**, accuracy **0.949** |
| **Weakest class** | white-glass (recall 0.792, F1 0.875); 22 of 106 missed, mostly as metal, plastic and brown-glass |
| **Grad-CAM shortcut check** | 6 images (clothes, shoes, paper, white-glass, plastic, metal): heat stays on the object, no background shortcut found. Only one image per class was checked, so this is not a full clearance. |
| **Binary metrics** | not used (12 classes); I analysed per-class recall instead |

### Limitations
Mostly web-scraped photos with one object on a simple background. Not tested on real conveyor-belt images or on dirty, crushed or wet items. It should not be used as an automatic sorter without human checking.

### Responsibility Questions
1. **Who could be harmed and how?** Plant workers and the recycling stream. Wrongly sorted glass, plastic or metal contaminates a batch. A missed battery can be a fire hazard.
2. **Where was the model not validated?** Real plant conditions: dirty, crushed, wet or overlapping items, bad lighting, and objects outside the 12 classes. It was not tested on other datasets.
3. **What is the human role?** A person checks low-confidence predictions and the weak classes (white-glass, metal, plastic), and removes hazardous items such as batteries by hand.
4. **What evidence would change my conclusions?** A big macro-F1 drop after masking the background, a much lower score on photos of waste around me or from a real sorting line, or Grad-CAM on more images showing heat on the background.
