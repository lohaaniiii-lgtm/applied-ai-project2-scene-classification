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


# Project 2, Week 7: Text and Multimodal AI

My Week 6 recycling model learned to **read**. This week I built and **measured** a text classifier, an entity extractor, tested regex fields and an OCR repair step, then combined the image model with a text note into **one masked, explainable JSON record**. Every component has a number, not just a claim.

## Results at a glance
| Component | What I built | Measured result |
|---|---|---|
| **Text classifier** | SMS Spam Collection, **5,572** texts, word vs character n-grams | macro-F1 **0.974** (word) vs **0.982** (char) |
| **Spam precision / recall** | same test set (149 spam) | word **0.979 / 0.933**, char **0.986 / 0.953** |
| **Top features** | read from the word model | spam: *call, txt, free, text, reply, claim*; ham: *my, me, ok, can*. Shortcuts: `uk` and `gt`/`lt` (HTML leftovers) |
| **NER, lab sentences** | spaCy `en_core_web_sm` vs gold set (16 entities) | F1 **0.467** → **0.812** with an entity ruler |
| **NER, my recycling sentences** | rules from sentences 1-5, scored on 6-10 | held-out F1 **0.400** → **0.593** |
| **Regex fields** | 6 lab patterns + my own: weight, plate, batch | **5 negative tests** passed (3 of my own) |
| **OCR repair** | CER and context rules | lab pairs **0.062 / 0.059 / 0.143** → **0.000**; clean invoice **0.000** |
| **Pipeline** | Week 6 ResNet-18 + text evidence | battery image P = **0.92** + 5 urgent terms → **REFER** |
| **Privacy** | CNIC, phone and email masked first | leak check returned **`[]`** |

## What I found
- **The character model won, but both text models missed Roman Urdu spam.** A normal Roman Urdu message with the words "free" and "call" scored **0.86 spam** with the word model and **0.32** with the character model.
- **Shortcuts hide in top features.** `uk` reflects where the data was collected, and `gt`/`lt` are HTML codes, not spam language.
- **Rules fix formats, not names.** spaCy tagged a Pakistani date as a place and a phone number as a date. An entity ruler fixed these, but on unseen sentences it found only **8 of 15** entities, so names need a model trained on my own data.
- **A regex that matched too much.** Without word boundaries my plate pattern returned `TCH-2026` from inside "BATCH". Adding `\b` fixed it.
- **When image and note disagree, the system still refers the case.** For a plastic photo with a note about a leaking battery, P(battery) was **0.009**, yet the note alone produced **REFER**. I recommend adding an explicit "image and note disagree" reason.

## Example record (masked)
The note contained a phone number and a CNIC, which never reach the record. Other image classes are cut for space.

```json
{
  "image": "battery1.jpg",
  "image_probabilities": {"battery": 0.919, "metal": 0.074, "brown-glass": 0.006},
  "note_masked": "Collected from Hanzala Traders in Rawalpindi on 05/11/2026. Old batteries, one swollen and leaking acid. Contact [PHONE], CNIC [CNIC].",
  "text_evidence": {
    "entities": [["Hanzala Traders", "ORG"], ["one", "CARDINAL"], ["CNIC", "ORG"], ["CNIC", "ORG"]],
    "fields": {"date": ["05/11/2026"]},
    "urgent_terms": ["batteries", "leak", "leaking", "acid", "swollen"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(battery) = 0.92", "note mentions: batteries, leak, leaking, acid, swollen"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

## Privacy
Personal identifiers are masked **before** any entity or field extraction, because extracting from the original note would copy the identifier into the record. I checked the final JSON for the phone digits and found none.

## Limitations
My held-out NER set has only **15 entities**, so one entity moves F1 by several points. The case records use **two examples** with notes I wrote, the OCR test image is **clean and synthetic**, and I did not check whether the example photos were in my Week 6 test set. This is **decision support only**, and a person decides.
