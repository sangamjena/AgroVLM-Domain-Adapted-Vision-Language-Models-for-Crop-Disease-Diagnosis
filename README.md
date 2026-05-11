# AgroVLM: Domain-Adapted Vision–Language Models for Crop Disease Diagnosis

This repository provides the **complete pipeline to reproduce the AgroVLM experiments**, including dataset preparation, instruction data generation, dataset conversion, and model fine-tuning.

**AgroVLM** is a suite of **domain-adapted vision–language models for crop disease diagnosis**. It currently includes two specialized sub-models:

- **BananaVLM** — trained for banana disease diagnosis
- **GroundnutVLM** — trained for groundnut (peanut) disease diagnosis

Both models are built on **LLaVA-v1.5-7B** and trained using **parameter-efficient LoRA fine-tuning**. The training pipeline is identical for both sub-models; only the dataset differs.

---

# Project Pipeline

The complete reproduction pipeline consists of the following stages (applicable to both BananaVLM and GroundnutVLM):

1. Download crop disease dataset
2. Generate instruction dataset (**BananaInstruct** / **GroundnutInstruct**)
3. Convert dataset to **LLaVA training format**
4. Fine-tune **LLaVA-v1.5-7B using LoRA**
5. Evaluate the trained model

---

# Repository Structure

```
agro-vlm
│
├── BananaVLM
│   ├── BananaInstruct
│   │   ├── attribute.zip
│   │   ├── external_knowledge.zip
│   │   ├── data_generation.py
│   │   └── data_format_converter.py
│   ├── dataset
│   │   └── banana.zip
│   │   └── readme.md
│   └── images
│
├── GroundnutVLM
│   ├── GroundnutInstruct
│   │   ├── attribute.zip
│   │   ├── external_knowledge.zip
│   │   ├── data_generation.py
│   │   └── data_format_converter.py
│   ├── dataset
│   │   └── groundnut.zip
│   │   └── readme.md
│   └── images
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

# 1. Clone the Repository

```bash
git clone https://github.com/samy101/agro-vlm.git
cd agro-vlm
```

---

# 2. Create Environment

Create a conda environment with Python 3.10.

```bash
conda create -n agrovlm python=3.10.19
conda activate agrovlm
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Download Dataset

## BananaVLM Dataset

The banana disease images are derived from the **Multi-Crop Disease Dataset**.

Original dataset:
https://data.mendeley.com/datasets/6243z8r6t6/1

Preprocessed banana subset:
https://drive.google.com/file/d/1AT8SL4yjpOBOxyyQB3CK-dSCJcssnRjv/view?usp=sharing

Download and extract inside `BananaVLM/`. Expected structure:

```
banana_images
├── bract_mosaic_virus
├── cordana
├── healthy
├── moko
├── panama
├── pestalotiopsis
├── sigatoka
└── yellow_and_black_sigatoka
```

**BananaVLM Dataset Statistics:**

| Disease                 | Images |
| ----------------------- | ------ |
| Bract Mosaic Virus      | 399    |
| Cordana                 | 558    |
| Moko                    | 445    |
| Panama                  | 1117   |
| Pestalotiopsis          | 573    |
| Sigatoka                | 1368   |
| Yellow & Black Sigatoka | 2791   |
| Healthy                 | 1079   |
| **Total**               | **8270** |

## GroundnutVLM Dataset

The groundnut disease images are similarly derived from a crop disease image dataset.

Download and extract inside `GroundnutVLM/`. Expected structure:

```
groundnut_images
├── alternaria_leaf_spot
├── cercospora_leaf_spot
├── healthy
├── leaf_spot
├── rust
└── web_blotch
```

The pipeline for GroundnutVLM is identical to BananaVLM — simply substitute the groundnut dataset and the corresponding `GroundnutInstruct` files in all subsequent steps.

---

# 5. Instruction Dataset Generation

The instruction dataset converts the original disease image dataset into a **multimodal instruction tuning dataset** suitable for training vision–language models. This process transforms **image-only data into structured conversational supervision** using a **three-stage automated pipeline**.

The pipeline enriches each image with:

- Visual symptom descriptions
- Multi-turn reasoning conversations
- Short classification-oriented question–answer pairs

This enables the model to learn disease recognition, symptom interpretation, contextual reasoning, and agricultural recommendations.

---

## Generation Pipeline

![Instruction Generation Pipeline](images/Slide1.jpg)

The pipeline consists of **three stages**.

---

## Stage 1 — Image-Grounded Symptom Description

**Model used:** `LLaVA-13B`

Each crop leaf image is processed by a multimodal LLM to generate **detailed natural language descriptions of visible symptoms**. These create a semantic bridge between the raw visual signal and agricultural terminology.

**Inputs per image:**
- Disease class label
- Plant type
- Healthy/diseased status

**Prompt Template:**

```
You are an agricultural assistant. Describe this image
of a {class label} from the {dataset name} dataset.
Based on the class/file name, this plant is classified as
{healthy status}. If diseased, clearly describe the visible symptoms.
If healthy, describe the normal plant characteristics.
```

**Example output:**

```
The image shows a banana leaf affected by Yellow and Black Sigatoka.
The leaf surface contains dark streaks and irregular black spots surrounded
by yellow halos. These lesions indicate fungal infection that reduces the
photosynthetic area of the plant.
```

---

## Stage 2 — Complex Agricultural Q&A Generation

**Model used:** `Mistral-7B`

Multi-turn conversational supervision is generated to teach the model **agricultural reasoning**. The LLM receives the Stage-1 image description, disease attributes, and curated external agricultural knowledge.

**Prompt Template:**

```
You are an AI assistant specialized in agricultural topics.
Generate exactly 3 to 5 Q&A pairs. Each question must begin with "Q:"
and each answer with "A:". Do not include any text outside Q&A.
Q&A should cover multi-turn and complex conversation.
Instructions:
• Focus on visible plant details.
• No speculation.
• No dataset references.
• No scientific names.
Context:
• Image Description: {description}
• Attributes: {attributes}
• External Knowledge: {external knowledge}
```

**Example output:**

```
Q: What disease could be affecting the banana plant?
A: The symptoms such as black streaks and yellow halos on the leaves suggest Yellow Sigatoka disease.

Q: How does this disease impact banana plants?
A: The disease reduces the photosynthetic area of the leaf, leading to reduced fruit yield.

Q: How does this disease spread?
A: It spreads through fungal spores carried by wind and rain splash.
```

---

## Stage 3 — Short Q&A Label Grounding

**Model used:** `Mistral-7B`

Short classification-oriented Q&A pairs are generated to strengthen the model's ability to **distinguish between visually similar diseases**. Answers are extremely short and deterministic.

**Prompt Template:**

```
You are a helpful agricultural tutor.
Generate 3–5 short Q&A pairs.
Rules:
• Start each question with "Q:" and answer with "A:".
• Answers must be single words only.
• No explanations.
Context:
• Image Description: {description}
• Attributes: {attributes}
• External Knowledge: {external knowledge}
```

**Example output:**

```
Q: What crop is shown in the image?
A: Banana

Q: Is the plant healthy or diseased?
A: Diseased

Q: What disease is visible?
A: Sigatoka
```

---

## Final Dataset

After all three stages, the generated supervision is stored in:

```
banana_disease.jsonl    # for BananaVLM
groundnut_disease.jsonl # for GroundnutVLM
```

Each file contains image path, image description, complex conversations, and simple classification Q&A.

| Sub-model     | Source Images | Q&A Pairs (approx.) |
| ------------- | ------------- | -------------------- |
| BananaVLM     | 8,270         | ~80,000              |
| GroundnutVLM  | TBD           | TBD                  |

---

# 6. Convert Dataset to LLaVA Format

Open `data_format_converter.py` and modify:

**For BananaVLM:**
```python
INPUT_FILE = "banana_disease.jsonl"
OUTPUT_FILE = "banana.json"
```

**For GroundnutVLM:**
```python
INPUT_FILE = "groundnut_disease.jsonl"
OUTPUT_FILE = "groundnut.json"
```

Run:

```bash
python data_format_converter.py
```

---

# 7. LLaVA Setup

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

---

# 8. Prepare Training Data

Copy the image folder and annotation file into `LLaVA/playground/data/`:

```
# BananaVLM
banana/
banana.json

# GroundnutVLM
groundnut/
groundnut.json
```

---

# 9. Configure Training Script

Open `scripts/v1_5/finetune_task_lora.sh` and update:

```
# BananaVLM
--data_path ./playground/data/banana.json
--image_folder ./playground/data/

# GroundnutVLM
--data_path ./playground/data/groundnut.json
--image_folder ./playground/data/
```

---

# 10. Training Configuration

Both BananaVLM and GroundnutVLM use identical LoRA fine-tuning settings:

| Parameter             | Value             |
| --------------------- | ----------------- |
| Base Model            | LLaVA-v1.5-7B     |
| Vision Encoder        | CLIP ViT-L/14-336 |
| LoRA Rank             | 16                |
| LoRA Alpha            | 32                |
| LoRA Dropout          | 0.05              |
| Epochs                | 5                 |
| Learning Rate         | 2e-5              |
| Batch Size            | 1                 |
| Gradient Accumulation | 4                 |
| Max Sequence Length   | 2048              |

**Hardware used:**

```
1 × NVIDIA RTX A6000 (48GB VRAM)
```

---

# 11. Run Training

```bash
bash scripts/v1_5/finetune_task_lora.sh
```

LoRA weights will be saved in `checkpoints/`.

---

# 12. Evaluation

Evaluation is performed on a **10% held-out test split**.

Two tasks are evaluated:

- **Identification** — Healthy vs. Diseased (binary classification)
- **Classification** — Fine-grained disease class prediction

Evaluation uses **substring matching** between predicted text and ground truth labels.

---

# Results

## BananaVLM Results

The table below compares **BananaVLM** against a range of recent open-source and closed-source vision–language models on the banana disease diagnosis benchmark.

**Tasks evaluated:**
- **Identification** — Binary prediction: Healthy vs. Diseased
- **Classification** — Fine-grained prediction across 8 disease classes

| Model                    | Identification (%) | Classification (%) |
| ------------------------ | :----------------: | :----------------: |
| **Open-source Models**   |                    |                    |
| LLaVA-7B                 | 67.40              | 21.20              |
| LLaVA-13B                | 68.95              | 19.57              |
| LLaVA-34B                | 80.44              | 22.08              |
| Qwen3-VL-8B              | 85.97              | 37.39              |
| Qwen3-VL-32B             | 85.52              | 36.89              |
| Qwen2.5-VL-72B           | 82.54              | 19.82              |
| LLaMA3-LLaVA-Next-8B     | 56.80              | 22.71              |
| Granite3.2-Vision        | 80.66              | 6.02               |
| Gemma3-4B                | 89.72              | 40.28              |
| Gemma3-12B               | 90.94              | 17.69              |
| MiniCPM-V                | 92.27              | 40.40              |
| **Closed-source Models** |                    |                    |
| Gemini 2.5 Flash Lite    | 90.01              | 22.98              |
| Gemini 2.5 Flash         | 91.22              | 40.87              |
| Gemini 2.5 Pro           | 93.22              | 42.42              |
| Gemini 3 Flash           | 93.90              | 40.08              |
| Gemini 3 Pro             | 94.90              | 41.79              |
| Gemini 3.1 Pro           | 92.90              | 42.28              |
| **BananaVLM (Ours)**     | **97.87**          | **93.54**          |

### Analysis

BananaVLM significantly outperforms all compared models on both tasks.

**Identification task:** General-purpose VLMs perform reasonably well here, with the strongest baseline being Gemini 3 Pro at 94.90%. BananaVLM surpasses this by nearly 3 percentage points, achieving **97.87%** — the highest identification accuracy across all evaluated models.

**Classification task:** This is where the gap is most dramatic. General-purpose models — even large ones like Qwen2.5-VL-72B (19.82%) and Gemma3-12B (17.69%) — struggle significantly with fine-grained disease differentiation. The best closed-source model, Gemini 2.5 Pro, achieves only 42.42%. BananaVLM more than doubles this, reaching **93.54%** — a gain of over 51 percentage points compared to the best general-purpose model.

This demonstrates the critical role of **domain-specific multimodal instruction tuning**: while large general VLMs can broadly detect disease presence, they lack the specialized visual grounding needed to distinguish between similar-looking banana diseases. BananaVLM's three-stage instruction generation pipeline — combining symptom descriptions, multi-turn agricultural reasoning, and short classification Q&A — provides the model with precisely the supervision required for fine-grained diagnosis.

---

## GroundnutVLM Results

Results for GroundnutVLM will be reported here upon completion of evaluation.

---

# References

- [LLaVA Official Repository](https://github.com/haotian-liu/LLaVA)
- [Ollama](https://ollama.com/)
- [Multi-crop Disease Dataset](https://data.mendeley.com/datasets/6243z8r6t6/1)
