# AgroVLM: Domain-Adapted Vision–Language Models for Crop Disease Diagnosis

> **Master of Technology Thesis** — Artificial Intelligence  
> Sangam Kumar Jena · Robert Bosch Centre for Cyber-Physical Systems, Indian Institute of Science, Bangalore · May 2026

---

## Abstract

Crop diseases are a major cause of agricultural yield loss worldwide, yet accurate field diagnosis remains challenging due to limited expert availability and the visual similarity between many disease classes. Although recent vision–language models (VLMs) have demonstrated strong general-purpose image understanding capabilities, they often struggle with specialised agricultural tasks requiring fine-grained disease recognition.

This work presents **AgroVLM**, a suite of domain-adapted vision–language models for crop disease diagnosis. AgroVLM currently consists of two specialised sub-models: **BananaVLM**, developed for banana disease diagnosis, and **GroundnutVLM**, developed for groundnut disease diagnosis. Both models are built on LLaVA-v1.5-7B and fine-tuned using parameter-efficient LoRA adaptation.

The results demonstrate that lightweight domain adaptation through automated instruction tuning is an effective and scalable strategy for specialised agricultural AI, enabling compact open-source VLMs to achieve strong performance on real-world crop disease diagnosis tasks — surpassing proprietary frontier models by up to **+73.9 percentage points** on classification.

---

## Table of Contents

- [Background](#background)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Limitations of Existing Approaches](#limitations-of-existing-approaches)
- [Proposed Approach — AgroVLM](#proposed-approach--agrovlm)
- [Contributions](#contributions)
- [Repository Structure](#repository-structure)
- [Project Pipeline](#project-pipeline)
- [Setup](#setup)
- [Datasets](#datasets)
- [Instruction Dataset Generation](#instruction-dataset-generation)
- [Model Architecture](#model-architecture)
- [Training Methodology](#training-methodology)
- [Training Configuration](#training-configuration)
- [Run Training](#run-training)
- [Evaluation Scripts](#evaluation-scripts)
- [Results — BananaVLM](#results--bananavlm)
- [Results — GroundnutVLM](#results--groundnutvlm)
- [LoRA vs DoRA Analysis](#lora-vs-dora-analysis)
- [Qualitative Evaluation (G-Eval)](#qualitative-evaluation-g-eval)
- [Human Expert Evaluation](#human-expert-evaluation)
- [Conclusion](#conclusion)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

---

## Background

Agriculture underpins the food security and livelihoods of billions of people worldwide, yet it remains acutely vulnerable to crop diseases. The Food and Agriculture Organization (FAO) estimates that plant diseases, pests, and pathogens are responsible for **20–40% of global crop losses** every year — a staggering toll that disproportionately impacts smallholder farmers and developing economies with the fewest resources to respond.

The challenge is not merely economic. As global population is projected to reach nearly 10 billion by 2050, sustaining sufficient agricultural output will require dramatically reducing these preventable losses. Yet the tools available to farmers in the field remain fundamentally unchanged: either wait for a trained agronomist to inspect the plant in person, or rely on experience and intuition. In large parts of sub-Saharan Africa, South Asia, and Southeast Asia — where banana and groundnut are critical staple and cash crops — there is roughly **one agricultural extension officer for every 1,000–3,000 farming households**. Timely expert diagnosis is simply not available at scale.

The consequences of delayed or incorrect diagnosis cascade quickly. A fungal infection identified a week late can spread to an entire field. A viral disease misidentified as a nutrient deficiency receives the wrong treatment. A farmer who cannot name the disease cannot source the correct fungicide, cannot follow quarantine protocols, and cannot report an outbreak to regional authorities. Each missed diagnosis is a compounding failure — agronomic, economic, and systemic.

>  **Banana** is the world's most consumed fruit and a dietary staple for over 400 million people, yet it is threatened by at least eight major diseases — including Fusarium wilt (Panama disease), Black Sigatoka, and Banana Bunchy Top Virus — many of which are visually similar in early stages and require expert differentiation.

>  **Groundnut** is a critical source of protein and income across Africa and South Asia, with major diseases like early leaf spot, late leaf spot, and rust causing yield losses of up to 70% in severe outbreaks if not caught early.

The global scale of this problem demands automated, accessible, and explainable diagnostic tools — ones that work in the field, on a mobile device, and without an agronomist standing next to the farmer.

---

## Motivation

Automated plant disease diagnosis is not a new idea. Computer vision has been applied to this problem for over a decade, and CNN-based classifiers trained on curated datasets like PlantVillage have achieved impressive accuracy numbers in controlled settings. But controlled accuracy and field utility are not the same thing.

Most deployed or published systems share a fundamental limitation: **they are pure classifiers**. Given an image, they output a label. They do not explain *why* they made that prediction, describe the visual symptoms they observed, discuss likely disease progression, or suggest a management strategy. A farmer who receives the output `"Panama Disease"` with no further context is not significantly better equipped than before — they still need an agronomist to translate that label into action.

This gap between classification accuracy and agronomic utility is the core motivation for this work. What farmers, extension officers, and crop diagnosticians actually need is not just a label — they need a *diagnostic response*: a description of what the model sees, a justification for the diagnosis, and practical guidance on what to do next.

Vision–language models (VLMs) are uniquely positioned to bridge this gap. By combining a visual encoder with a large language model backbone, VLMs can generate free-form natural language responses grounded in image content — responses that can describe symptoms, explain reasoning, and recommend management actions in plain language. This is precisely the form of output that has real utility in an agricultural advisory context.

However, the general-purpose VLMs available today — whether open-source models like LLaVA, Qwen-VL, and Gemma, or frontier closed-source systems like GPT-4o and Gemini — are trained on internet-scale data that contains almost no specialised agricultural pathology content. They can describe a plant in broad terms, but they cannot reliably identify fine-grained disease classes, distinguish visually similar conditions, or produce the kind of confident, specific, agronomically grounded response that a real diagnostic tool requires. The domain gap is simply too large for zero-shot generalisation to close.

**The motivation for AgroVLM is therefore twofold:** to bring the *format* of VLM responses (natural language, explanatory, actionable) to a domain where it is acutely needed, and to close the domain gap through targeted fine-tuning so that the model's outputs are accurate enough to be trusted.

---

## Problem Statement

> **Can a compact, open-source vision–language model be domain-adapted to perform accurate, explainable crop disease diagnosis — surpassing both unimodal classifiers in utility and general-purpose VLMs in accuracy — using only automated instruction tuning without manual annotation?**

More concretely, this work addresses the following requirements:

**1. Fine-grained visual discrimination.** Crop disease diagnosis requires distinguishing between classes that are visually very similar — for example, early leaf spot versus late leaf spot in groundnut, or Sigatoka versus Black Sigatoka in banana. The model must be able to make these distinctions reliably, not just identify broad categories.

**2. Explainable, language-grounded output.** The diagnostic output must go beyond a class label. It must describe the visible symptoms in the image, ground the diagnosis in those observations, and communicate useful information about disease progression and management — in natural language that a non-expert can act on.

**3. Out-of-domain generalisation.** A model that works only on images from the same distribution as its training data has limited practical value. The model must generalise to images captured under different lighting, at different growth stages, with different camera hardware, and from geographically distinct populations of the same disease.

**4. Scalability without expert annotation.** Manual annotation of agricultural datasets is expensive and requires rare domain expertise. A practical pipeline must be able to generate high-quality instruction-tuning data at scale from image datasets alone, using automation rather than human labelling.

**5. Efficiency.** The model must be trainable and deployable on accessible hardware — a single GPU rather than a multi-node cluster — to lower the barrier to replication and extension to new crops.

---

## Limitations of Existing Approaches

### Unimodal Classifiers (CNNs, ViTs)

The dominant paradigm for automated crop disease diagnosis has been supervised CNN or Vision Transformer classification on labelled image datasets. While these models can achieve high top-1 accuracy on held-out test sets, they have several structural limitations for real-world deployment:

- **Label-only outputs.** A classifier outputs a disease label and a confidence score. It cannot describe symptoms, explain its reasoning, recommend treatment, or answer follow-up questions. The diagnostic utility is therefore fundamentally limited — the label is a starting point, not an answer.

- **Closed vocabulary.** A classifier trained on a fixed set of classes cannot handle novel diseases, ambiguous presentations, or multi-label scenarios (e.g., a plant with both a fungal infection and a nutrient deficiency). It will always force the image into one of its training classes, even when none fits.

- **No grounding in visual evidence.** There is no mechanism for a standard classifier to communicate *which* visual features drove its prediction. Explainability methods like Grad-CAM produce heatmaps, but these are post-hoc approximations, not natural-language explanations that can be communicated to a farmer.

- **Poor distribution shift robustness.** CNNs trained on curated datasets notoriously degrade under distribution shift — changes in lighting, camera angle, growth stage, or geographic region. Without explicit augmentation or domain randomisation, generalisation to truly out-of-distribution images is fragile.

### General-Purpose Vision–Language Models (Zero-Shot)

Recent large VLMs such as GPT-4o, Gemini, LLaVA-34B, Qwen2.5-VL-72B, and Gemma3 represent a significant step forward in multimodal understanding. They can describe images, answer questions, and generate natural language — but they consistently underperform on specialised agricultural tasks for the following reasons:

- **Severe domain gap.** Internet-scale training data contains very little content on crop pathology, and what exists is rarely at the level of granularity required for fine-grained disease classification. These models can identify "a sick-looking plant" but rarely distinguish between, say, Cordana and Pestalotiopsis on a banana leaf.

- **Inconsistent label grounding.** General-purpose VLMs frequently produce responses that describe symptoms accurately but assign the wrong disease name, or produce correct labels with incorrect symptom descriptions. The connection between visual evidence and label is not reliably grounded.

- **No agronomic knowledge integration.** Standard VLMs do not have systematically integrated knowledge of disease progression, management protocols, fungicide recommendations, or environmental risk factors. Their management advice is generic and often inaccurate.

- **Scale does not compensate for domain specificity.** Our experiments show that even the largest tested baseline — Qwen2.5-VL-72B — achieves only 8.95% classification accuracy on the banana in-domain test set and 17.05% out-of-domain. A 10× smaller domain-adapted model achieves 92.21% and 83.28% respectively. **Parameter count is not a substitute for domain knowledge.**

| Approach | Disease Label | Symptom Description | Management Guidance | Out-of-Domain |
|---|---|---|---|---|
| CNN Classifier | Accurate | None | None | Fragile |
| General-Purpose VLM (zero-shot) | Often wrong | Generic | Generic | Degraded |
| **AgroVLM (ours)** | Accurate | Specific & grounded | Agronomically relevant | Robust |

---

## Proposed Approach — AgroVLM

AgroVLM addresses all of the above limitations through three integrated design choices:

**1. Vision–Language model foundation.** By building on LLaVA-v1.5-7B — which combines a CLIP ViT-L/14 vision encoder with a Vicuna-7B language model backbone — AgroVLM inherits strong visual understanding and fluent language generation. The model can describe what it sees, not just classify it.

**2. Automated domain-specific instruction tuning.** Rather than relying on manual annotation or zero-shot prompting, AgroVLM is trained on a rich instruction dataset generated automatically from raw disease image collections. The three-stage pipeline produces approximately 80,000 Q&A pairs for banana and 75,000 for groundnut — covering symptom description, multi-turn agronomic reasoning, and classification label grounding — without any human labelling of training data.

**3. Parameter-efficient LoRA fine-tuning.** Full fine-tuning of a 7B model is impractical on a single GPU. LoRA adaptation introduces fewer than 1% additional trainable parameters while achieving convergence that full fine-tuning cannot on this task. Crucially, both the LoRA weights on the language model and the MLP projection layer are trained, allowing the visual-to-language alignment to adapt alongside the language generation.

The result is a model that is simultaneously more *accurate* than any tested baseline (open-source or proprietary) on disease classification, and more *useful* in the sense that matters agronomically — providing grounded symptom descriptions and actionable management guidance that human experts consistently prefer by a margin of ~98.6%.

---

## Contributions

1. **AgroVLM** — A suite of domain-adapted VLMs for crop disease diagnosis, comprising BananaVLM and GroundnutVLM, both built on LLaVA-v1.5-7B with LoRA fine-tuning.
2. **Automated instruction dataset generation pipeline** — A three-stage pipeline (BananaInstruct / GroundnutInstruct) that converts raw crop disease image datasets into rich multimodal instruction-tuning datasets (~80,000 and ~75,000 Q&A pairs respectively) without manual annotation, using LLaVA-1.5-13B and Mistral-7B.
3. **Comprehensive evaluation** — Benchmarking against 13 open-source and 7 closed-source VLM baselines using in-domain and out-of-domain test sets, G-Eval scoring by 9 LLM judges, and blind human expert evaluation.
4. **State-of-the-art performance** — BananaVLM and GroundnutVLM substantially outperform all evaluated baselines on both disease identification and fine-grained classification tasks.

---

## Repository Structure

```
agro-vlm
│
├── Attribute/                        # Per-disease attribute text files for instruction generation
├── BananaInstruct/                   # Generated instruction dataset for BananaVLM
├── Dora Evaluation/                  # DoRA model evaluation results and logs
├── External Knowledge/               # Curated agronomic knowledge used in Stage 2 generation
├── GroundnutInstruct/                # Generated instruction dataset for GroundnutVLM
├── Qualitative Evaluation/           # G-Eval judge outputs and human expert comparison files
├── Quantitative Evaluation/          # Accuracy, F1, and per-class classification reports
│
├── classification.py                 # LoRA model — fine-grained disease classification evaluation
├── identification.py                 # LoRA model — binary healthy/diseased identification evaluation
├── dora_classification.py            # DoRA model — fine-grained disease classification evaluation
├── dora_identification.py            # DoRA model — binary healthy/diseased identification evaluation
├── open_source_model_compare.py      # Zero-shot evaluation of open-source VLM baselines via Ollama
├── qualitative_eval.py               # Generates paired base vs. fine-tuned outputs for qualitative review
├── geval.py                          # G-Eval: multi-judge LLM scoring across 4 agronomic dimensions
│
├── data_format_converter.py          # Converts JSONL instruction data to LLaVA training JSON format
├── data_generation.py                # Three-stage instruction dataset generation pipeline
├── dataset_overview.pdf              # Visual overview of dataset composition
├── dataset_overview-1.png
├── dora_plot_bananavlm.png           # LoRA vs DoRA performance curve — BananaVLM
├── dora_plot_groundnut.png           # LoRA vs DoRA performance curve — GroundnutVLM
├── lora_epoch_banana.png             # Epoch-wise accuracy curve — BananaVLM
├── lora_epoch_groundnut.png          # Epoch-wise accuracy curve — GroundnutVLM
├── sample-images.pdf
├── sample-images-1.png               # Sample crop disease images from the dataset
├── Slide1.jpg                        # Three-stage instruction generation pipeline diagram
├── requirements.txt
└── README.md
```

---

## Project Pipeline

The complete reproduction pipeline consists of the following stages (identical for both BananaVLM and GroundnutVLM; only the dataset differs):

1. Download crop disease dataset
2. Generate instruction dataset (**BananaInstruct** / **GroundnutInstruct**) via the three-stage pipeline
3. Convert dataset to **LLaVA training format**
4. Fine-tune **LLaVA-v1.5-7B using LoRA**
5. Evaluate the trained model (in-domain + out-of-domain)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/samy101/agro-vlm.git
cd agro-vlm
```

### 2. Create Environment

```bash
conda create -n agrovlm python=3.10.19
conda activate agrovlm
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Datasets

### BananaVLM Dataset

The banana disease images are derived from the **Multi-Crop Disease Dataset** (Mendeley Data).

- **Original dataset:** https://data.mendeley.com/datasets/6243z8r6t6/1  
- **Preprocessed banana subset:** https://drive.google.com/file/d/1AT8SL4yjpOBOxyyQB3CK-dSCJcssnRjv/view?usp=sharing

<p align="center">
  <img src="dataset_overview-1.png" width="700"/>
</p>

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

**BananaVLM Dataset Statistics**

| Disease / Condition | # Images | Disease / Condition | # Images |
|---------------------|----------:|---------------------|----------:|
| Bract Mosaic Virus  | 399       | Sigatoka            | 1,368     |
| Cordana             | 558       | Yellow Band Sigatoka | 2,791   |
| Insect Pest         | 688       | Pestalotiopsis      | 573       |
| Moko                | 445       | Healthy             | 1,079     |
| Panama              | 1,117     | **Total**           | **9,018** |

The dataset exhibits moderate class imbalance, with Yellow & Black Sigatoka being the most represented class and Bract Mosaic Virus the least. A **10% held-out split (905 images)** is reserved for in-domain evaluation.

---

### GroundnutVLM Dataset

The groundnut disease dataset contains four disease classes and one healthy class collected under real-world agricultural conditions.

- **Dataset:** https://drive.google.com/drive/folders/1yyeKwa_3Z_khbMaWFnIvkvNO2b6vERi4?usp=sharing
- **Out-of-domain test set:** [Kaggle — Groundnut Plant Leaf Data](https://www.kaggle.com/datasets/warcoder/groundnut-plant-leaf-data)

<p align="center">
  <img src="sample-images-1.png" width="700"/>
</p>

Download and extract inside `GroundnutVLM/`. Expected structure:

```
groundnut_images
├── early_leaf_spot
├── late_leaf_spot
├── healthy
├── rust
└── nutritional_deficiency
```

**GroundnutVLM Dataset Statistics:**

| Disease Class          | Images   |
|------------------------|----------|
| Early Leaf Spot        | 1,213    |
| Late Leaf Spot         | 1,984    |
| Nutritional Deficiency | 1,663    |
| Rust                   | 3,195    |
| Healthy                | 409      |
| **Total**              | **8,464** |

A **10% held-out split (849 images)** is used for in-domain evaluation. The out-of-domain evaluation set (1,000 images) is sourced from a separate Kaggle dataset not seen during training.

---

### Test Set Statistics

#### In-Domain Test Sets (10% held-out)

| Crop      | Class                   | Images | Total |
|-----------|-------------------------|--------|-------|
| Banana    | Bract Mosaic Virus      | 40     | 797   |
|           | Cordana                 | 56     |       |
|           | Insect Pest             | 69     |       |
|           | Moko                    | 45     |       |
|           | Panama                  | 112    |       |
|           | Pestalotiopsis          | 58     |       |
|           | Sigatoka                | 137    |       |
|           | Yellow & Black Sigatoka | 280    |       |
| Groundnut | Early Leaf Spot         | 122    | 849   |
|           | Healthy                 | 41     |       |
|           | Late Leaf Spot          | 199    |       |
|           | Nutritional Deficiency  | 167    |       |
|           | Rust                    | 320    |       |

#### Out-of-Domain Test Sets (independent sources)

| Crop      | Class                   | Images | Total |
|-----------|-------------------------|--------|-------|
| Banana    | Pestalotiopsis          | 173    | 743   |
|           | Cordana                 | 162    |       |
|           | Yellow & Black Sigatoka | 90     |       |
|           | Insect Pest             | 86     |       |
|           | Healthy                 | 86     |       |
|           | Moko                    | 55     |       |
|           | Bract Mosaic Virus      | 50     |       |
|           | Panama                  | 41     |       |
| Groundnut | Early Leaf Spot         | 168    | 1,000 |
|           | Healthy                 | 168    |       |
|           | Nutritional Deficiency  | 167    |       |
|           | Late Leaf Spot          | 165    |       |
|           | Rust                    | 332    |       |

---

## Instruction Dataset Generation

The instruction generation pipeline transforms image-only crop disease datasets into rich multimodal instruction-tuning datasets. The pipeline consists of **three stages**, each targeting a different aspect of agricultural language supervision.

![Three-stage instruction generation pipeline](Slide1.jpg)

---

### Stage 1 — Agricultural Image Description Generation

**Model used:** LLaVA-1.5-13B

Each crop disease image is processed to generate detailed symptom descriptions grounded in observable visual features — lesion morphology, chlorosis, necrosis, streak formation, tissue collapse — while avoiding speculation or unsupported claims.

**Prompt Template:**

```python
f"You are an agricultural assistant. "
f"Describe visible symptoms in this image of a {crop} plant "
f"affected by {disease}. Focus only on what is visible."
```

**Example output:**

```
The image shows a banana leaf with elongated dark streaks and irregular
black lesions surrounded by yellow discoloration. Several portions of the
leaf appear necrotic and dried, indicating severe leaf damage.
```

---

### Stage 2 — Complex Agricultural Q&A Generation

**Model used:** Mistral-7B

Multi-turn conversational supervision is generated using the Stage 1 image description, disease attributes, crop information, and curated external agricultural knowledge sourced from university resources, extension service documents, technical blogs, and research articles. Between 3 and 5 Q&A pairs are generated per image, teaching symptom interpretation, contextual reasoning, disease progression understanding, and management recommendations.

**Prompt Template:**

```python
f"""
You are an AI assistant specialized in agricultural topics.

Generate exactly 3 to 5 Q&A pairs.
Each question must begin with "Q:" and each answer with "A:".
Do not include any text outside Q&A.

Instructions:
- Focus on visible plant details.
- No speculation.
- No dataset references.
- No scientific names.

Context:
Crop: {data['crop']}
Disease: {data['disease']}
Image Description: {data["description"]}
Attributes: {data["attributes"]}
External Knowledge: {external_knowledge}
"""
```

**Example output:**

```
Q: What disease symptoms are visible on the leaf?
A: The leaf shows dark streaks, yellow discoloration, and necrotic lesions.

Q: How does this disease affect the plant?
A: The damaged leaf tissue reduces the photosynthetic area and weakens plant growth.

Q: What environmental conditions favor this disease?
A: High humidity and prolonged leaf wetness promote disease development.
```

---

### Stage 3 — Short Q&A Label Grounding

**Model used:** Mistral-7B

Short classification-oriented Q&A pairs are generated to strengthen the model's label grounding and improve discrimination between visually similar disease classes. Answers are intentionally single-word and deterministic.

**Prompt Template:**

```python
"""
You are a helpful agricultural tutor.

Generate 3–5 short Q&A pairs.

Rules:
- Start each question with "Q:" and answer with "A:".
- Answers must be single words only.
- No explanations.

Context:
Crop: {data['crop']}
Disease: {data['disease']}
"""
```

**Example output:**

```
Q: What crop is shown?
A: Banana

Q: Is the plant healthy?
A: Diseased

Q: What disease is present?
A: Sigatoka

Q: What plant part is affected?
A: Leaf
```

---

### Final Instruction Datasets

Each generated record (stored in JSONL format) contains: image path, Stage 1 description, Stage 2 conversational supervision, and Stage 3 short Q&A pairs. A random subset per disease class was manually reviewed by agricultural domain experts for factual and agronomic correctness. Generation was executed on an NVIDIA RTX 5090 GPU.

| Sub-model        | Source Images | Q&A Pairs (approx.) |
|------------------|---------------|----------------------|
| BananaInstruct   | 8,270         | ~80,000              |
| GroundnutInstruct| 8,464         | ~75,000              |

---

### Convert Dataset to LLaVA Format

Open `data_format_converter.py` and modify:

```python
# BananaVLM
INPUT_FILE = "banana_disease.jsonl"
OUTPUT_FILE = "banana.json"

# GroundnutVLM
INPUT_FILE = "groundnut_disease.jsonl"
OUTPUT_FILE = "groundnut.json"
```

Then run:

```bash
python data_format_converter.py
```

---

## Model Architecture

Both BananaVLM and GroundnutVLM are built on **LLaVA-v1.5-7B**, which combines:

- **Vision Encoder:** CLIP ViT-L/14 at 336×336 resolution — encodes input images into visual tokens.
- **Projection Layer:** A two-layer MLP that maps visual tokens into the language model's embedding space.
- **Language Model:** Vicuna-7B (LLaMA-2 based) — generates natural language responses conditioned on visual tokens and the instruction prompt.

The 7B parameter scale makes the model tractable for single-GPU LoRA fine-tuning while providing a strong foundation for visual instruction following.

---

## Training Methodology

### Parameter-Efficient Fine-Tuning

Full fine-tuning of a 7B parameter model requires prohibitive GPU memory. AgroVLM therefore uses **Parameter-Efficient Fine-Tuning (PEFT)** methods that freeze pretrained weights and introduce a small number of trainable parameters into attention layers. Two variants were explored:

#### Low-Rank Adaptation (LoRA)

LoRA introduces trainable low-rank decomposition matrices into frozen attention layers. For a pretrained weight matrix W₀ ∈ ℝ^(d×k):

```
W = W₀ + ΔW = W₀ + BA
```

where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k) are trainable, r ≪ min(d, k) is the LoRA rank, and W₀ remains frozen. Effective trainable parameter count is r(d + k) per layer — orders of magnitude smaller than full fine-tuning.

#### Weight-Decomposed Low-Rank Adaptation (DoRA)

DoRA decomposes W₀ into a magnitude component m and direction component V, updating direction via LoRA and magnitude as a separate trainable scalar. While DoRA more closely mimics full fine-tuning behaviour in theory, **experiments showed it does not converge for the multi-class classification task** — LoRA was selected for the final models.

### LLaVA Setup

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

### Prepare Training Data

Copy the image folder and annotation file into `LLaVA/playground/data/`:

```
# BananaVLM
banana/
banana.json

# GroundnutVLM
groundnut/
groundnut.json
```

### Configure Training Script

Open `scripts/v1_5/finetune_task_lora.sh` and update:

```bash
# BananaVLM
--data_path ./playground/data/banana.json
--image_folder ./playground/data/

# GroundnutVLM
--data_path ./playground/data/groundnut.json
--image_folder ./playground/data/
```

---

## Training Configuration

Both BananaVLM and GroundnutVLM use the following LoRA fine-tuning settings. LoRA/DoRA weights are applied exclusively to the attention layers of Vicuna-7B; the CLIP vision encoder and MLP projection head are **not frozen** — the projector is fine-tuned at the same learning rate to adapt the visual-to-language mapping to the agricultural domain. All training uses **DeepSpeed ZeRO-2** optimisation for memory efficiency on a single GPU.

| Parameter             | Value                        |
|-----------------------|------------------------------|
| Base Model            | LLaVA-v1.5-7B                |
| Vision Encoder        | CLIP ViT-L/14-336            |
| LoRA Rank (r)         | 64                           |
| LoRA Alpha (α)        | 16                           |
| LoRA Dropout          | 0.05                         |
| Training Epochs       | 7                            |
| Learning Rate         | 2 × 10⁻⁵                    |
| Projector LR          | 2 × 10⁻⁵                    |
| Batch Size            | 1                            |
| Gradient Accumulation | 2 (effective batch size: 2)  |
| LR Scheduler          | Cosine                       |
| Warmup Ratio          | 0.03                         |
| Weight Decay          | 0.01                         |
| Max Sequence Length   | 1,024 tokens                 |
| Precision             | BF16 + TF32                  |
| Optimiser             | DeepSpeed ZeRO-2             |
| Hardware              | 1× NVIDIA RTX A6000 (48 GB)  |

---

## Run Training

```bash
bash scripts/v1_5/finetune_task_lora.sh
```

LoRA weights will be saved in `checkpoints/`.

---

## Evaluation Scripts

The repository provides six evaluation scripts covering all model variants, tasks, and baseline comparisons. Before running any script, update the `MODEL_PATH`, `MODEL_BASE`, and `DATASET_DIR` variables to match your local paths.

---

### `classification.py` — LoRA Model: Fine-Grained Disease Classification

Evaluates a **LoRA fine-tuned AgroVLM** (BananaVLM or GroundnutVLM) on the **disease classification task** — predicting the specific disease class from an image.

The model is prompted with the list of valid class names and asked to return one label. Predictions are matched against ground truth using substring matching and fuzzy overlap (SequenceMatcher), which tolerates minor wording variations in generated outputs. Outputs a per-class scikit-learn classification report (precision, recall, F1), confusion matrix, AUC score, and latency/throughput metrics. Results are saved to a timestamped CSV in `./crop_reports/`.

```bash
# Edit MODEL_PATH and DATASET_DIR, then run:
python classification.py
```

---

### `identification.py` — LoRA Model: Binary Disease Identification

Evaluates a **LoRA fine-tuned AgroVLM** on the **binary identification task** — determining whether a plant is healthy or diseased.

The model is given the prompt `"Is this crop diseased or not? Answer only yes or no."` and the response is mapped to `disease` (yes) or `healthy` (no). Computes accuracy, precision, recall, F1, and the full confusion matrix (TP/TN/FP/FN). Tracks per-image latency and GPU peak memory usage. Progress is checkpointed every 5 images to CSV.

```bash
# Edit MODEL_PATH and DATASET_DIR, then run:
python identification.py
```

---

### `dora_classification.py` — DoRA Model: Fine-Grained Disease Classification

Equivalent to `classification.py` but for the **DoRA fine-tuned** model variant.

Uses the lower-level LLaVA model builder API (`load_pretrained_model` + `PeftModel.from_pretrained`) instead of the higher-level `eval_model` wrapper, which is required because DoRA weights include a `non_lora_trainables.bin` that must be loaded separately before the PEFT adapter. The model is loaded once per crop and reused across all images in that crop's test set for efficiency.

```bash
# Edit MODEL_PATH and DATASET_DIR, then run:
python dora_classification.py
```

---

### `dora_identification.py` — DoRA Model: Binary Disease Identification

Equivalent to `identification.py` but for the **DoRA fine-tuned** model variant.

Also uses the low-level model builder with explicit `non_lora_trainables.bin` loading. Unlike the LoRA identification script, this version reloads the model from scratch for each individual image — a design choice from early-stage debugging that was preserved to isolate any per-image memory issues during DoRA experiments.

```bash
# Edit MODEL_PATH and DATASET_DIR, then run:
python dora_identification.py
```

---

### `open_source_model_compare.py` — Zero-Shot Baseline Evaluation via Ollama

Evaluates **open-source VLM baselines** (e.g., LLaVA-7B/13B/34B, Qwen2.5-VL, Qwen3-VL, Gemma3, MiniCPM-V, Granite, BakLLaVA) on both tasks using the **Ollama API** for local inference.

Images are base64-encoded and sent to `http://localhost:11434/api/generate`. For the identification task (`task="id"`), responses are parsed for yes/no. For the classification task (`task="cls"`), responses are matched against ground truth disease names using fuzzy matching (substring overlap + SequenceMatcher). Accuracy results are appended to `accuracy.txt` and per-image predictions are saved to `results/<model_name>_<task>_results.csv`.

Requires [Ollama](https://ollama.com/) to be running locally with the target models pulled.

```bash
# Start Ollama and pull models first, then:
python open_source_model_compare.py
```

---

### `qualitative_eval.py` — Paired Output Generation for Qualitative Review

Generates **side-by-side response pairs** from the base model and the LoRA fine-tuned model for 200 balanced samples drawn from the test set. Outputs a CSV (`banana_eval_results.csv`) and a Word document report (`banana_report.docx`) with the input image, ground truth label, and both models' free-form responses to the prompt `"see the image and tell me what type of disease, visible symptoms and possible solution?"`.

The Word report is used directly as input for the human expert blind evaluation protocol — evaluators review pairs without knowing which response came from which model.

```bash
python qualitative_eval.py
```

---

### `geval.py` — G-Eval: Multi-Judge LLM Scoring

Implements the **G-Eval evaluation framework** to assess response quality across four agronomic dimensions using nine open-source LLM judges served via Ollama.

For each image in the qualitative evaluation CSV, all nine judges independently score both the base model response and the AgroVLM response on a 0–5 scale across: disease identification, classification accuracy, visible symptoms, and management strategy. Each judge also declares an overall winner (A or B). Scores are averaged per judge and win rates are computed. Results are saved to `multi_llm_comparison_groundnut.csv` and a formatted text summary to `final_llm_results_groundnut.txt`.

Requires all nine judge models to be available via the local Ollama instance.

```bash
python geval.py
```

---

## Results — BananaVLM

### Epoch-wise Performance (LoRA)

<p align="center">
  <img src="lora_epoch_banana.png" width="700"/>
</p>

| Epoch   | In-Domain Classification (%) | Out-of-Domain Classification (%) | In-Domain Identification (%) | Out-of-Domain Identification (%) |
|---------|:----------------------------:|:--------------------------------:|:----------------------------:|:--------------------------------:|
| Epoch 3 | 63.61                        | 62.40                            | 97.67                        | 99.73                            |
| Epoch 5 | 91.31                        | 65.75                            | 97.23                        | 99.60                            |
| Epoch 7 | **92.31**                    | **83.28**                        | 96.90                        | 98.38                            |

Classification accuracy improves substantially with more training epochs. Identification accuracy remains consistently high across all epochs (>96%), indicating binary disease detection converges early while fine-grained classification continues to refine.

---

### In-Domain Results (Epoch 7)

#### Per-Class Classification Report

| Class                   | Precision | Recall | F1-Score | Support |
|-------------------------|:---------:|:------:|:--------:|:-------:|
| Bract Mosaic Virus      | 0.97      | 0.95   | 0.96     | 40      |
| Bunchy Top Insect Pest  | 0.90      | 0.44   | 0.59     | 61      |
| Cordana                 | 0.96      | 0.93   | 0.95     | 56      |
| Moko                    | 1.00      | 0.96   | 0.98     | 45      |
| Panama                  | 0.95      | 0.91   | 0.93     | 112     |
| Pestalotiopsis          | 1.00      | 0.88   | 0.94     | 58      |
| Sigatoka                | 0.99      | 0.99   | 0.99     | 137     |
| Yellow & Black Sigatoka | 0.95      | 0.99   | 0.97     | 280     |
| **Overall Accuracy**    |           |        |          | **92.21%** |

Most disease classes achieve F1-scores above 0.93. The lowest-performing class is **Bunchy Top Insect Pest** (F1: 0.59), driven by low recall (0.44) — likely due to lower representation (69 training images) and visual overlap with healthy leaf tissue.

#### Identification Report (In-Domain)

| Metric    | Value  |
|-----------|:------:|
| Accuracy  | 96.91% |
| Precision | 0.9661 |
| Recall    | **1.0000** |
| F1-Score  | 0.9827 |

The model achieves **perfect recall (1.0)** — it correctly flags every diseased plant as diseased. This is a critical property for a crop disease detection system where false negatives (missed disease) are far more costly than false positives.

---

### Out-of-Domain Results (Epoch 7)

#### Per-Class Classification Report

| Class                          | Precision | Recall | F1-Score | Support |
|--------------------------------|:---------:|:------:|:--------:|:-------:|
| Cordana                        | 0.9933    | 0.9198 | 0.9551   | 162     |
| Pestalotiopsis                 | 0.9568    | 0.7688 | 0.8526   | 173     |
| Banana Moko                    | 0.9483    | 1.0000 | 0.9735   | 55      |
| Banana Panama                  | 0.7400    | 0.9024 | 0.8132   | 41      |
| Banana Bract Mosaic Virus      | 0.6712    | 0.9800 | 0.7967   | 50      |
| Banana Yellow & Black Sigatoka | 0.5732    | 1.0000 | 0.7287   | 90      |
| Banana Insect Pest             | 1.0000    | 0.2532 | 0.4040   | 79      |
| **Overall Accuracy**           |           |        |          | **83.28%** |

The most challenging OOD class is **Banana Insect Pest** (F1: 0.40, recall: 0.25), reflecting difficulty in recognising visually variable insect damage patterns that differ from training images. Cordana and Moko show strong generalisation (F1: 0.96, 0.97).

#### Identification Report (Out-of-Domain)

| Metric    | Value  |
|-----------|:------:|
| Accuracy  | 98.38% |
| Precision | 0.9864 |
| Recall    | 0.9954 |
| F1-Score  | 0.9909 |

---

### Open-Source VLM Comparison — BananaVLM

| Model                  | In-Domain ID (%) | In-Domain Cls (%) | OOD ID (%) | OOD Cls (%) |
|------------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| LLaVA-7B               | 67.40            | 12.15             | 27.55      | 9.59        |
| LLaVA-13B              | 68.95            | 10.72             | 7.61       | 20.85       |
| LLaVA-34B              | 80.44            | 8.73              | 79.15      | 17.35       |
| Qwen2.5-VL-7B          | 71.05            | 0.00              | 90.72      | 21.92       |
| Qwen3-VL-8B            | 85.97            | 12.04             | 99.24      | 21.46       |
| Qwen3-VL-32B           | 85.52            | 9.72              | 99.70      | 18.57       |
| Qwen2.5-VL-72B         | 82.54            | 8.95              | 98.63      | 17.05       |
| LLaMA3-LLaVA-Next-8B   | 56.80            | 11.16             | 14.61      | 10.96       |
| Granite3.2-Vision      | 80.66            | 0.11              | 2.44       | 31.81       |
| Gemma3-4B              | 89.72            | 12.49             | 100.00     | 14.92       |
| Gemma3-12B             | 90.94            | 4.86              | 99.70      | 15.22       |
| LLaVA-Phi3-3.8B        | 17.57            | 2.32              | 11.87      | 27.70       |
| MiniCPM-V              | 92.27            | 11.60             | 43.68      | 23.29       |
| **BananaVLM (Ours)**   | **96.90**        | **92.21**         | **98.38**  | **83.28**   |

### Closed-Source VLM Comparison — BananaVLM

| Model                    | In-Domain ID (%) | In-Domain Cls (%) | OOD ID (%) | OOD Cls (%) |
|--------------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| Gemini 2.5 Flash Lite    | 90.01            | 22.98             | 97.14      | 13.24       |
| Gemini 2.5 Flash         | 91.22            | 40.87             | 93.36      | 14.76       |
| Gemini 2.5 Pro           | 93.22            | 42.42             | 95.01      | 20.09       |
| Gemini 3 Flash Preview   | 93.90            | 40.08             | 96.44      | 23.90       |
| Gemini 3.1 Flash Lite    | 92.90            | 42.28             | 98.10      | 13.85       |
| **BananaVLM (Ours)**     | **96.90**        | **92.21**         | **98.38**  | **83.28**   |

On the in-domain benchmark, BananaVLM outperforms all open-source and closed-source baselines by a wide margin on classification. The best closed-source model (Gemini 2.5 Pro) reaches 42.42% — BananaVLM surpasses it by **+49.8 percentage points** in-domain and **+63.3 pp** OOD.

---

## Results — GroundnutVLM

### Epoch-wise Performance (LoRA)

<p align="center">
  <img src="lora_epoch_groundnut.png" width="700"/>
</p>

| Epoch   | ID — In-Domain | ID — OOD  | Cls — In-Domain | Cls — OOD  |
|---------|:--------------:|:---------:|:---------------:|:----------:|
| Epoch 3 | 96.70%         | 96.90%    | 97.90%          | 99.04%     |
| Epoch 5 | 98.94%         | 97.00%    | 98.27%          | 98.32%     |
| Epoch 7 | **99.18%**     | **97.10%**| **98.27%**      | **99.40%** |

GroundnutVLM achieves remarkably high accuracy from very early in training. OOD classification reaches **99.40%** at Epoch 7 — the highest single result in this work.

---

### In-Domain Results (Epoch 7)

#### Per-Class Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|:---------:|:------:|:--------:|:-------:|
| Early Leaf Spot        | 0.95      | 0.98   | 0.97     | 122     |
| Late Leaf Spot         | 0.99      | 0.97   | 0.98     | 199     |
| Nutritional Deficiency | 1.00      | 0.91   | 0.95     | 167     |
| Rust                   | 1.00      | 0.96   | 0.98     | 320     |
| **Overall Accuracy**   |           |        | **0.97** | **98.27%** |

All four disease classes achieve F1 ≥ 0.95. Nutritional Deficiency has the lowest recall (0.91), reflecting visual ambiguity between mild deficiency symptoms and early-stage disease lesions.

#### Identification Report (In-Domain)

| Metric             | Value  |
|--------------------|:------:|
| Accuracy           | 99.18% |
| Weighted Precision | 0.99   |
| Weighted Recall    | 1.00   |

---

### Out-of-Domain Results (Epoch 7)

#### Per-Class Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|:---------:|:------:|:--------:|:-------:|
| Early Leaf Spot        | 1.0000    | 1.0000 | **1.0000** | 168   |
| Late Leaf Spot         | 1.0000    | 1.0000 | **1.0000** | 165   |
| Nutritional Deficiency | 1.0000    | 0.9701 | 0.9848   | 167     |
| Rust                   | 1.0000    | 0.9759 | 0.9878   | 332     |
| **Overall Accuracy**   |           |        |          | **99.40%** |

Early Leaf Spot and Late Leaf Spot achieve **perfect precision, recall, and F1 (1.0)** on an entirely unseen dataset. Weighted precision is perfect (1.0000) — zero false positives across the entire OOD test set.

#### Identification Report (Out-of-Domain)

| Metric    | Value  |
|-----------|:------:|
| Accuracy  | 97.10% |

---

### Open-Source VLM Comparison — GroundnutVLM

| Model                     | In-Domain ID (%) | In-Domain Cls (%) | OOD ID (%) | OOD Cls (%) |
|---------------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| LLaVA-7B                  | 40.75            | 4.59              | 42.90      | 5.30        |
| LLaVA-13B                 | 18.73            | 8.01              | 25.90      | 8.40        |
| LLaVA-34B                 | 80.33            | 14.02             | 72.20      | 12.20       |
| Qwen2.5-VL-7B             | 83.75            | 1.77              | 80.70      | 1.60        |
| Qwen3-VL-8B               | 95.05            | 34.16             | 92.80      | 35.40       |
| Qwen3-VL-32B              | 95.41            | 28.27             | 93.00      | 27.90       |
| Qwen2.5-VL-72B            | 90.58            | 3.53              | 87.20      | 2.40        |
| Granite3.2-Vision         | 9.31             | 10.37             | 16.70      | 7.70        |
| Gemma3-4B                 | 96.00            | 32.63             | 84.70      | 31.60       |
| Gemma3-12B                | 97.06            | 42.29             | 93.30      | 44.20       |
| LLaVA-Phi3-3.8B           | 16.02            | 6.12              | 20.80      | 6.80        |
| MiniCPM-V                 | 35.81            | 28.74             | 35.70      | 25.40       |
| BakLLaVA                  | 21.55            | 1.65              | 26.30      | 2.00        |
| **GroundnutVLM (Ours)**   | **99.18**        | **98.27**         | **97.10**  | **99.40**   |

### Closed-Source VLM Comparison — GroundnutVLM

| Model                    | In-Domain ID (%) | In-Domain Cls (%) | OOD ID (%) | OOD Cls (%) |
|--------------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| Gemini 2.5 Flash Lite    | 96.70            | 4.95              | 96.70      | 12.10       |
| Gemini 2.5 Flash         | 98.94            | 11.26             | 97.00      | 24.80       |
| Gemini 2.5 Pro           | **99.18**        | 24.38             | 97.10      | 41.60       |
| Gemini 3 Flash Preview   | 98.35            | 17.95             | 96.90      | 26.40       |
| Gemini 3.1 Flash Lite    | 97.76            | 24.38             | 96.80      | 23.40       |
| **GroundnutVLM (Ours)**  | **99.18**        | **98.27**         | **97.10**  | **99.40%** |

GroundnutVLM matches the best closed-source model (Gemini 2.5 Pro) on identification (both 99.18%) while achieving **4× higher classification accuracy** in-domain (98.27% vs. 24.38%) and exceeding the best OOD closed-source result by **+57.8 pp** (99.40% vs. 41.60%).

---

## LoRA vs DoRA Analysis

<p align="center">
  <img src="dora_plot_bananavlm.png" width="700"/>
</p>
<p align="center">
  <img src="dora_plot_groundnut.png" width="700"/>
</p>

Both PEFT variants were trained with identical hyperparameters (rank 64, alpha 16) for direct comparison.

| Method | Banana In-Domain Cls (%) | Banana OOD Cls (%) | Groundnut In-Domain Cls (%) | Groundnut OOD Cls (%) |
|--------|--------------------------|--------------------|-----------------------------|-----------------------|
| LoRA   | **92.21**                | **83.28**          | **98.27**                   | **99.40**             |
| DoRA   | ~19.5 (degrades)         | ~9.0               | ~38–40 (saturates)          | ~38–40                |

**DoRA does not converge for multi-class disease classification.** While identification accuracy remains high (>96%) under both methods, DoRA's magnitude–direction weight decomposition introduces optimisation instability that impairs multi-class classification. Small errors in the magnitude component amplify misclassifications across visually similar disease classes. **LoRA is used for all final models.**

---

## Qualitative Evaluation (G-Eval)

Beyond raw accuracy, response quality is assessed using the **G-Eval prompting strategy**: nine open-source LLM judges independently score responses on a 1–5 scale across four agronomic dimensions, with the Win Rate representing the fraction of pairwise comparisons where the fine-tuned model is preferred.

| Dimension            | Description |
|----------------------|-------------|
| **Disease ID**       | Correctness and specificity of disease identification |
| **Classification**   | Accuracy of disease class assignment |
| **Symptom Description** | Quality and precision of described visual symptoms |
| **Management**       | Relevance and accuracy of suggested treatment or management actions |

### BananaVLM G-Eval Results

| Judge Model    | Base Disease ID | BananaVLM Disease ID | Base Cls | BananaVLM Cls | Base Symptoms | BananaVLM Symptoms | Base Mgmt | BananaVLM Mgmt | Win Rate |
|----------------|:---------------:|:--------------------:|:--------:|:-------------:|:-------------:|:------------------:|:---------:|:--------------:|:--------:|
| llama3.1       | 1.47            | 4.28                 | 1.87     | 4.45          | 2.86          | 4.63               | 2.87      | 4.61           | 0.92     |
| mistral        | 1.16            | 4.72                 | 1.13     | 4.46          | 2.60          | 4.23               | 2.79      | 3.97           | 0.83     |
| mistral-small  | 0.72            | 3.70                 | 0.80     | 3.57          | 2.17          | 3.95               | 2.93      | 3.75           | 0.89     |
| gemma3:12b     | 1.17            | 3.70                 | 1.01     | 3.61          | 2.03          | 3.68               | 2.25      | 3.70           | 0.96     |
| qwen2.5:14b    | 0.81            | 3.87                 | 0.55     | 3.56          | 1.48          | 3.86               | 2.75      | 3.81           | 0.90     |
| qwen2.5:32b    | 1.27            | 3.53                 | 0.98     | 3.30          | 1.95          | 3.57               | 3.03      | 3.66           | 0.82     |
| qwen2.5:7b     | 1.33            | 3.68                 | 1.07     | 3.18          | 2.18          | 4.01               | 2.47      | 3.58           | 0.81     |
| qwen2.5:72b    | 1.19            | 3.56                 | 0.62     | 3.20          | 2.14          | 3.66               | 2.89      | 3.85           | 0.87     |
| mixtral        | 1.90            | 4.37                 | 1.45     | 4.27          | 2.43          | 4.63               | 2.33      | 4.37           | 0.96     |

BananaVLM is preferred in **81–96%** of comparisons across all nine judges and all four dimensions.

### GroundnutVLM G-Eval Results

| Judge Model    | Base Disease ID | GroundnutVLM Disease ID | Base Cls | GroundnutVLM Cls | Base Symptoms | GroundnutVLM Symptoms | Base Mgmt | GroundnutVLM Mgmt | Win Rate |
|----------------|:---------------:|:-----------------------:|:--------:|:----------------:|:-------------:|:---------------------:|:---------:|:-----------------:|:--------:|
| Qwen2.5-7B     | 1.62            | 4.12                    | 1.49     | 3.88             | 2.52          | 4.43                  | 2.62      | 4.00              | 0.89     |
| Qwen2.5-14B    | 1.37            | 4.50                    | 0.80     | 4.42             | 2.23          | 4.49                  | 3.43      | 4.43              | 0.94     |
| Qwen2.5-32B    | 1.66            | 4.46                    | 1.10     | 4.42             | 2.36          | 4.30                  | 3.47      | 4.09              | 0.93     |
| Qwen2.5-72B    | 1.69            | 4.47                    | 0.81     | 4.43             | 2.71          | 4.18                  | 3.33      | 4.35              | 0.97     |
| Mixtral        | 2.34            | 4.64                    | 1.60     | 4.52             | 2.68          | 4.74                  | 2.36      | 4.73              | 0.99     |
| Mistral        | 1.80            | 4.73                    | 1.83     | 4.67             | 3.39          | 4.35                  | 3.23      | 3.88              | 0.86     |
| Mistral-Small  | 0.85            | 4.44                    | 0.95     | 4.28             | 2.50          | 4.10                  | 3.19      | 4.00              | 0.99     |
| Llama3.1       | 1.58            | 4.68                    | 1.97     | 4.72             | 3.18          | 4.95                  | 2.63      | 4.67              | 0.97     |
| Gemma3-12B     | 1.30            | 4.20                    | 0.95     | 4.13             | 2.12          | 3.97                  | 2.28      | 3.88              | **1.00** |

GroundnutVLM win rates range from **0.86 to 1.00** across all nine judges.

---

## Human Expert Evaluation

Five domain experts independently reviewed pairwise comparisons between base model and AgroVLM outputs in a **blind evaluation protocol** (no model labels shown). Evaluators selected the response they considered more accurate, complete, and agronomically actionable.

### BananaVLM (198 comparisons per evaluator)

| Evaluator   | Base Model | BananaVLM  | Win Rate   |
|-------------|:----------:|:----------:|:----------:|
| Evaluator 1 | 1 / 198    | 197 / 198  | 99.49%     |
| Evaluator 2 | 0 / 198    | 198 / 198  | **100.00%**|
| Evaluator 3 | 7 / 198    | 191 / 198  | 96.46%     |
| Evaluator 4 | 4 / 198    | 194 / 198  | 97.98%     |
| Evaluator 5 | 3 / 198    | 195 / 198  | 98.48%     |
| **Aggregate** | **15/990** | **975/990** | **98.48%** |

### GroundnutVLM (200 comparisons per evaluator)

| Evaluator   | Base Model | GroundnutVLM | Win Rate   |
|-------------|:----------:|:------------:|:----------:|
| Evaluator 1 | 0 / 200    | 200 / 200    | **100.00%**|
| Evaluator 2 | 2 / 200    | 198 / 200    | 99.00%     |
| Evaluator 3 | 0 / 200    | 200 / 200    | **100.00%**|
| Evaluator 4 | 6 / 200    | 194 / 200    | 97.00%     |
| Evaluator 5 | 4 / 200    | 196 / 200    | 98.00%     |
| **Aggregate** | **12/1000** | **988/1000** | **98.8%** |

Across all 1,990 total expert judgements, AgroVLM was preferred in **98.6%** of comparisons. The near-perfect agreement across five independent evaluators, closely corroborated by G-Eval win rates of 0.81–1.00 across nine LLM judges, provides strong convergent validity for the qualitative improvements introduced by domain fine-tuning.

---

## Conclusion

AgroVLM demonstrates that **lightweight, targeted domain adaptation through automated instruction tuning is an effective and scalable strategy for agricultural AI**. Key findings:

- **LoRA substantially outperforms DoRA** for multi-class disease classification (92.21% vs. ~19.5% on banana; 98.27% vs. ~39% on groundnut).
- **Strong out-of-domain generalisation:** GroundnutVLM achieves 99.40% OOD classification with zero false positives; BananaVLM retains 83.28% on an entirely independent test set.
- **Domain adaptation decisively outperforms scale:** AgroVLM exceeds Gemini 2.5 Pro by up to +73.9 pp on classification and surpasses all open-source baselines by more than 50 pp — regardless of model size. A 7B-parameter model with targeted fine-tuning on ~800 labelled images outperforms a zero-shot 72B model.
- **Expert validation confirms practical utility:** Human agronomists preferred AgroVLM outputs in 98.6% of 1,990 blind comparisons, closely corroborated by automated G-Eval judges.

---

## Limitations and Future Work

- **Additional crops:** Extending the pipeline to wheat, rice, tomato, and other staple crops would broaden agronomic coverage.
- **Multilingual deployment:** Adapting for multilingual instruction tuning and on-device deployment (e.g. 4-bit quantisation) to increase accessibility in developing regions.
- **Improving weak classes:** Bunchy Top Insect Pest (banana, F1=0.59) and Insect Pest OOD (F1=0.40) require targeted augmentation or few-shot adaptation strategies.
- **Continual learning:** A continual learning framework would allow incremental updates as new disease variants emerge without full retraining.
- **Field deployment:** Integration into a mobile application with real-time inference, GPS-tagged disease reporting, and agronomist feedback loops.

---

## References

- [LLaVA Official Repository](https://github.com/haotian-liu/LLaVA)
- [Ollama](https://ollama.com/)
- [Multi-Crop Disease Dataset (Mendeley)](https://data.mendeley.com/datasets/6243z8r6t6/1)
- Liu et al. (2023). Visual Instruction Tuning. *NeurIPS.*
- Liu et al. (2024). Improved Baselines with Visual Instruction Tuning. *CVPR.*
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR.*
- Liu et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *arXiv:2402.09353.*
- Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP.*
- Rajbhandari et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC20.*
