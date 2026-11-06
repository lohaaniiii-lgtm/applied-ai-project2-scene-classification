# Project 2: Deep Learning Applications

**Name:** Muhammad Luqman Khan | **Course:** Introduction to Applied AI

This repo holds all of Project 2: Week 5 (transfer learning), Week 6 (recycling waste sorting), Week 7 (text and multimodal AI) and Lab 8 (the FastAPI service).

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

---

## Project 2, Week 7: Text and Multimodal AI

My Week 6 recycling model learned to **read**. This week I built and **measured** a text classifier, an entity extractor, tested regex fields and an OCR repair step, then combined the image model with a text note into **one masked, explainable JSON record**. Every component has a number, not just a claim.

### Results at a glance
| Component | What I built | Measured result |
|---|---|---|
| **Text classifier** | SMS Spam Collection, **5,572** texts, word vs character n-grams | macro-F1 **0.974** (word) vs **0.982** (char) |
| **Spam precision / recall** | same test set (149 spam) | word **0.979 / 0.933**, char **0.986 / 0.953** |
| **Top features** | read from the word model | spam: *call, txt, free, text, reply, claim*; ham: *my, me, ok, can*. Shortcuts: `uk` and `gt`/`lt` (HTML leftovers) |
| **NER, lab sentences** | spaCy `en_core_web_sm` vs gold set (16 entities) | F1 **0.467** → **0.812** with an entity ruler (I changed the names in the provided sentences, so this is **not comparable** with the shared reference 0.533 → 0.875) |
| **NER, my recycling sentences** | rules from sentences 1-5, scored on 6-10 | held-out F1 **0.400** → **0.593** |
| **Regex fields** | 6 lab patterns + my own: weight, plate, batch | **5 negative tests** passed (3 of my own) |
| **OCR repair** | CER and context rules | lab pairs **0.062 / 0.059 / 0.143** → **0.000**; clean synthetic invoice **0.000** (no real errors, so the rules were not truly tested) |
| **Pipeline** | Week 6 ResNet-18 + text evidence | battery image P = **0.92** + 5 urgent terms → **REFER** |
| **Privacy** | CNIC, phone and email masked first | leak check returned **`[]`** |

### What I found
- **The character model won, but both text models missed Roman Urdu spam.** A normal Roman Urdu message with the words "free" and "call" scored **0.86 spam** with the word model and **0.32** with the character model.
- **Shortcuts hide in top features.** `uk` reflects where the data was collected, and `gt`/`lt` are HTML codes, not spam language.
- **Rules fix formats, not names.** spaCy tagged a Pakistani date as a place and a phone number as a date. An entity ruler fixed these, but on unseen sentences it found only **8 of 15** entities, so names need a model trained on my own data.
- **A regex that matched too much.** Without word boundaries my plate pattern returned `TCH-2026` from inside "BATCH". Adding `\b` fixed it.
- **When image and note disagree, the system still refers the case.** For a plastic photo with a note about a leaking battery, P(battery) was **0.009**, yet the note alone produced **REFER**. I recommend adding an explicit "image and note disagree" reason.

### Example record (masked)
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

### Privacy
Personal identifiers are masked **before** any entity or field extraction, because extracting from the original note would copy the identifier into the record. I checked the final JSON for the phone digits and found none.

### Limitations
My held-out NER set has only **15 entities**, so one entity moves F1 by several points. The case records use **two examples** with notes I wrote, the OCR test image is **clean and synthetic**, and I did not check whether the example photos were in my Week 6 test set. This is **decision support only**, and a person decides.


## Lab 8: Recycling Waste Sorting API (Vision + Text)

A FastAPI web service that serves my Week 6 image model (ResNet-18, 12 waste classes, macro-F1 0.928)
and my Week 7 text evidence (PII masking, field extraction, urgent-word detection).
The positive class that must not be missed is **battery** (hazardous waste).

### Files
| File | Purpose |
|---|---|
| `app.py` | `/health`, `/predict`, `/analyze` endpoints |
| `test_api.py` | 8 automated tests |
| `bench.py` | latency benchmark |
| `sample.jpeg` | test image (a battery photo) used by the tests and the benchmark |
| `week6_model.pt` | trained Week 6 checkpoint (not committed if large) |

### How to run
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs and use "Try it out".

### Settings changed for my project
- `POSITIVE = 'battery'`
- `URGENT = battery, batteries, leak, leaking, acid, hazardous, fire, smoke, swollen, sharp, broken glass`
- `MODEL_PATH = week6_model.pt`
- Added email masking and a weight pattern in `PATTERNS` (from Week 7).

### API usage
| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | none | status, model_loaded, 12 classes, positive class |
| `POST /predict` | image (JPEG/PNG, max 5 MB) | label, probabilities, latency_ms |
| `POST /analyze` | image + text note (max 2,000 chars) | prediction, masked note, fields, urgent terms, decision, reasons |

```
curl -X POST http://127.0.0.1:8000/predict -F "file=@sample.jpeg;type=image/jpeg"

curl -X POST http://127.0.0.1:8000/analyze ^
     -F "file=@sample.jpeg;type=image/jpeg" ^
     -F "note=Old batteries, swollen and leaking acid. Call 0312-5558899, CNIC 35202-1234567-1"
```
Result: `/predict` on my battery sample returned `label: battery` with P(battery) = 0.9295 (latency 121.2 ms on that call).
`/analyze` returned the note with `[PHONE]` and `[CNIC]` in place of the real numbers,
urgent terms (batteries, leaking, acid, swollen) and `decision: REFER`.
The phone number and CNIC do not appear anywhere in the response.

Error codes: 415 wrong file type, 413 file over 5 MB, 400 unreadable image, 422 missing file or note too long,
401 missing API key (only when `API_KEY` is set).

### Test results
```
pytest -q test_api.py
........                                                                 [100%]
8 passed, 1 warning in 11.77s
```
Tests cover: health, response schema and probabilities summing to 1, parity with calling the model directly,
wrong file type (415), too-large file (413), corrupt image (400), missing file (422), and PII masking.
The warning is only an httpx deprecation notice from Starlette's test client.

### Latency and capacity
Hardware: Intel Core i5-6200U @ 2.30 GHz (2 cores, 4 threads), 8 GB RAM, Windows, CPU-only PyTorch, one uvicorn worker.
```
python bench.py
n=50  p50=70 ms  p95=97 ms  max=259 ms
Throughput with 1 worker ~ 13.2 requests/s
```
The 259 ms maximum is most likely the first request while the server warms up.

**Little's law:** one worker handles one request at a time, so rate = 1 / p50 = 1 / 0.070 s = 14.3 requests/s.
Per day: 14.3 x 86,400 s is about **1.23 million requests/day** (86,400 / 0.070 = 1,234,286).
The measured mean gives 13.2 req/s, about 1.14 million per day.

### Bad inputs tried
- I sent `app.py` as the upload to `/predict` with content type `text/plain`. The API returned
  HTTP 415 `{"detail":"Upload a JPEG or PNG image"}`, with no crash and no stack trace.
- I created a 6 MB file (`fsutil file createnew big.jpg 6000000`) and sent it to `/predict`. The API returned
  HTTP 413 `{"detail":"Image larger than 5 MB"}`, again with no stack trace.

### Check Your Understanding
1. **A client uploads a PDF to /predict. Which status code, and why not 500?**
   415 Unsupported Media Type. The mistake is the client's, and 4xx codes mean a client error. A 500 means a bug in my server.
   The API checks the content type before decoding, so a bad upload is rejected cleanly instead of crashing.
2. **Why is the model loaded in lifespan, not inside the endpoint?**
   Loading the checkpoint takes seconds and hundreds of MB of memory. In `lifespan` it happens once when the server starts.
   Inside the endpoint, every request would reload it, making every call slow and wasting memory.
3. **p50 = 40 ms and p95 = 400 ms: what does it tell you, and what do you check?**
   A typical request is fast, but the slowest 5% are about 10 times slower. That is a tail-latency problem, so the average hides it.
   I would check: queueing when requests overlap, blocking code in the event loop, cold starts, large images,
   CPU contention from other programs, and the max value and per-request timings in the logs.
4. **500,000 requests/day, peak 5 times the average, 25 requests/s per worker. How many workers?**
   Average = 500,000 / 86,400 = 5.8 req/s. Peak = 5 x 5.8 = 28.9 req/s.
   Workers = 28.9 / 25 = 1.16, rounded up to **2 workers**. I would use 3 for headroom.
5. **Why can model inference inside an `async def` endpoint slow other requests?**
   An async endpoint runs on the single event loop. Heavy CPU work like inference does not yield,
   so it blocks the loop and every other request waits, even `/health`. Using `run_in_threadpool` moves the work
   to a worker thread so the loop stays free.
6. **How many communication channels does a 4-person team have?**
   n(n-1)/2 = 4 x 3 / 2 = **6 channels**.

### Reflection
Serving the model showed that a good test score is not enough. Preprocessing must match training (the parity test),
bad input must fail safely (415, 413, 400, 422), and personal data must be masked before anything is returned.
The p50 of 70 ms and about 1.2 million requests per day per worker are enough for a recycling-collection use case on a laptop.
Limits: the model was trained on web photos, so real-world lighting and clutter may lower accuracy,
and the output is decision support only; a human decides.
