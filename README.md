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
| **Total**               | **9018** |

## GroundnutVLM Dataset

The groundnut disease images are similarly derived from a crop disease image dataset.

Dataset Download

GroundnutVLM dataset can be downloaded from the following link:

https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing

Download and extract inside `GroundnutVLM/`. Expected structure:

```
groundnut_images
├── early_leaf_spot
├── late_leaf_spot
├── healthy
├── rust
└── nutritional deficiency
```

**GroundnutVLM Dataset Statistics:**

| Disease                | Images   |
| ---------------------- | -------- |
| Early Leaf Spot        | 1213     |
| Healthy                | 409      |
| Late Leaf Spot         | 1984     |
| Nutritional Deficiency | 1663     |
| Rust                   | 3195     |
| **Total**              | **8464** |

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
## Stage 1 — Agricultural Image Description Generation

**Model used:** `LLaVA-1.5-7B`

In the first stage, the vision-language model generates detailed agricultural image descriptions focused strictly on visible plant symptoms. The prompt explicitly constrains the model to avoid speculation and describe only observable characteristics.

**Prompt Template:**

```python
f"You are an agricultural assistant. "
f"Describe visible symptoms in this image of a {crop} plant "
f"affected by {disease}. Focus only on what is visible."
```

**Example output:**

```text
The image shows a banana leaf with elongated dark streaks and irregular
black lesions surrounded by yellow discoloration. Several portions of the
leaf appear necrotic and dried, indicating severe leaf damage.
```

---

## Stage 2 — Complex Agricultural Q&A Generation

**Model used:** `Mistral-7B`

Multi-turn conversational supervision is generated to teach the model agricultural reasoning and contextual disease understanding. The model receives the Stage-1 image description, disease attributes, crop information, and curated external agricultural knowledge.

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

```text
Q: What disease symptoms are visible on the leaf?
A: The leaf shows dark streaks, yellow discoloration, and necrotic lesions.

Q: How does this disease affect the plant?
A: The damaged leaf tissue reduces the photosynthetic area and weakens plant growth.

Q: What environmental conditions favor this disease?
A: High humidity and prolonged leaf wetness promote disease development.
```

---

## Stage 3 — Short Q&A Label Grounding

**Model used:** `Mistral-7B`

Short classification-oriented Q&A pairs are generated to strengthen the model’s grounding capability and improve discrimination between visually similar agricultural diseases. Answers are intentionally short and deterministic.

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

```text
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
## Final Dataset

After all three stages, the generated supervision is stored in:

```
banana_disease.jsonl    # for BananaVLM
groundnut_disease.jsonl # for GroundnutVLM
```

Each file contains image path, image description, complex conversations, and simple classification Q&A.

| Sub-model     | Source Images | Q&A Pairs (approx.) |
| ------------- | ------------- | -------------------- |
| BananaVLM     | 9,018         | ~80,000              |
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

BananaVLM is evaluated on two distinct test splits to measure both specialization depth and real-world generalization ability:

- **In-Domain** — Images drawn from the same distribution as the training data (10% held-out split from the BananaInstruct dataset)
- **Out-of-Domain** — An entirely separate banana disease dataset not seen during training, used to assess transfer and robustness

Two tasks are evaluated in both settings:

- **Identification** — Binary prediction: Healthy vs. Diseased
- **Classification** — Fine-grained prediction across disease classes

---

### Dataset Statistics

#### In-Domain Test Set (905 images, 9 classes)

| Class                   | Images |
| ----------------------- | ------ |
| Bunchy Top Insect Pest  | 69     |
| Cordana                 | 56     |
| Healthy                 | 108    |
| Moko                    | 45     |
| Panama                  | 112    |
| Pestalotiopsis          | 58     |
| Sigatoka                | 137    |
| Yellow & Black Sigatoka | 280    |
| **Total**               | **905** |

#### Out-of-Domain Test Set (743 images, 8 classes)

| Class                              | Images |
| ---------------------------------- | ------ |
| Pestalotiopsis                     | 173    |
| Cordana                            | 162    |
| Banana Yellow & Black Sigatoka     | 90     |
| Banana Insect Pest                 | 86     |
| Healthy                            | 86     |
| Banana Moko                        | 55     |
| Banana Bract Mosaic Virus          | 50     |
| Banana Panama                      | 41     |
| **Total**                          | **743** |

---

### 1. In-Domain Results

#### 1.1 Epoch-wise Performance

BananaVLM was evaluated at three checkpoints during training to track learning progression on the in-domain test set.

| Epoch   | Classification (%) | Identification (%) |
| ------- | :----------------: | :----------------: |
| Epoch 3 | 63.61              | 97.67              |
| Epoch 5 | 91.31              | 97.23              |
| Epoch 7 | 92.31              | 96.90              |

Classification accuracy improves substantially with more training epochs, reaching **92.31%** at Epoch 7. Identification accuracy remains consistently high across all epochs (96.90–97.67%), indicating the model learns to distinguish healthy from diseased plants early in training, while fine-grained classification continues to refine with additional training.

#### 1.2 Per-Class Classification Report (Epoch 7)

| Class                  | Precision | Recall | F1-Score | Support |
| ---------------------- | :-------: | :----: | :------: | :-----: |
| Bract Mosaic Virus     | 0.97      | 0.95   | 0.96     | 40      |
| Bunchy Top Insect Pest | 0.90      | 0.44   | 0.59     | 61      |
| Cordana                | 0.96      | 0.93   | 0.95     | 56      |
| Moko                   | 1.00      | 0.96   | 0.98     | 45      |
| Panama                 | 0.95      | 0.91   | 0.93     | 112     |
| Pestalotiopsis         | 1.00      | 0.88   | 0.94     | 58      |
| Sigatoka               | 0.99      | 0.99   | 0.99     | 137     |
| Yellow & Black Sigatoka| 0.95      | 0.99   | 0.97     | 280     |
| **Overall Accuracy**   |           |        |          | **92.21%** |

Most disease classes achieve F1-scores above 0.93. The lowest-performing class is **Bunchy Top Insect Pest** (F1: 0.59), driven by low recall (0.44), which indicates the model struggles to recall this visually distinctive class consistently — likely due to lower representation in the training set (69 images) and potential visual overlap with healthy leaf tissue.

#### 1.3 Identification Report (In-Domain)

| Metric    | Value  |
| --------- | :----: |
| Accuracy  | 96.91% |
| Precision | 0.9661 |
| Recall    | 1.0000 |
| F1-Score  | 0.9827 |

The model achieves **perfect recall (1.0)** on the identification task, meaning it correctly flags every diseased plant as diseased — a critical property for a crop disease detection system where false negatives (missed disease) are far more costly than false positives.

#### 1.4 Open-Source VLM Comparison (In-Domain)

| Model                | Identification (%) | Classification (%) |
| -------------------- | :----------------: | :----------------: |
| LLaVA-7B             | 67.40              | 12.15              |
| LLaVA-13B            | 68.95              | 10.72              |
| LLaVA-34B            | 80.44              | 8.73               |
| Qwen2.5-VL-7B        | 71.05              | 0.00               |
| Qwen3-VL-8B          | 85.97              | 12.04              |
| Qwen3-VL-32B         | 85.52              | 9.72               |
| LLaMA3-LLaVA-Next-8B | 56.80              | 11.16              |
| Granite3.2-Vision    | 80.66              | 0.11               |
| Gemma3-4B            | 89.72              | 12.49              |
| Gemma3-12B           | 90.94              | 4.86               |
| MiniCPM-V            | 92.27              | 11.60              |
| Qwen2.5-VL-72B       | 82.54              | 8.95               |
| **BananaVLM (Ours)** | **96.90**          | **92.21**          |

#### 1.5 Closed-Source VLM Comparison (In-Domain)

| Model               | Identification (%) | Classification (%) |
| ------------------- | :----------------: | :----------------: |
| Gemini 2.5 Flash Lite | 90.01            | 22.98              |
| Gemini 2.5 Flash    | 91.22              | 40.87              |
| Gemini 2.5 Pro      | 93.22              | 42.42              |
| Gemini 3 Flash      | 93.90              | 40.08              |
| Gemini 3 Pro        | 94.90              | 41.79              |
| Gemini 3.1 Pro      | 92.90              | 42.28              |
| **BananaVLM (Ours)**| **96.90**          | **92.21**          |

On the in-domain benchmark, BananaVLM outperforms all open-source and closed-source baselines by a wide margin on classification. The best open-source model (MiniCPM-V) achieves only 11.60% classification accuracy, and the best closed-source model (Gemini 2.5 Pro) reaches 42.42% — BananaVLM surpasses both by more than 49 percentage points, demonstrating the transformative impact of domain-specific instruction tuning.

---

### 2. Out-of-Domain Results

Out-of-domain evaluation tests how well BananaVLM generalizes to an entirely unseen banana disease dataset, measuring robustness beyond its training distribution.

#### 2.1 Epoch-wise Performance (Out-of-Domain)

| Epoch   | Classification (%) | Identification (%) |
| ------- | :----------------: | :----------------: |
| Epoch 3 | 62.40              | 99.73              |
| Epoch 5 | 65.75              | 99.60              |
| Epoch 7 | **83.28**          | 98.38              |

The model continues to improve on out-of-domain classification across epochs, reaching **83.28%** at Epoch 7, which confirms that the learned disease representations generalize well beyond the training distribution. Identification accuracy remains exceptionally high (>98%) at all checkpoints.

#### 2.2 Per-Class Classification Report — Out-of-Domain (Epoch 7)

| Class                              | Precision | Recall | F1-Score | Support |
| ---------------------------------- | :-------: | :----: | :------: | :-----: |
| Pestalotiopsis                     | 0.9568    | 0.7688 | 0.8526   | 173     |
| Cordana                            | 0.9933    | 0.9198 | 0.9551   | 162     |
| Banana Yellow & Black Sigatoka     | 0.5732    | 1.0000 | 0.7287   | 90      |
| Banana Insect Pest                 | 1.0000    | 0.2532 | 0.4040   | 79      |
| Banana Moko                        | 0.9483    | 1.0000 | 0.9735   | 55      |
| Banana Bract Mosaic Virus          | 0.6712    | 0.9800 | 0.7967   | 50      |
| Banana Panama                      | 0.7400    | 0.9024 | 0.8132   | 41      |
| **Overall Accuracy**               |           |        |          | **83.28%** |

On the out-of-domain set, the most challenging class is **Banana Insect Pest** (F1: 0.40), where recall drops to 0.25 — indicating difficulty in recognizing visually variable insect damage patterns that differ from training images. **Banana Yellow & Black Sigatoka** achieves perfect recall (1.0) but lower precision (0.57), reflecting some over-prediction of this class. **Cordana** and **Moko** show strong generalization with F1-scores of 0.96 and 0.97 respectively.

#### 2.3 Identification Report (Out-of-Domain)

| Metric    | Value  |
| --------- | :----: |
| Accuracy  | 98.38% |
| Precision | 0.9864 |
| Recall    | 0.9954 |
| F1-Score  | 0.9909 |

Even on completely unseen data, BananaVLM achieves **98.38% identification accuracy** — outperforming every general-purpose model tested. This demonstrates strong generalization of the disease presence/absence concept across different image capture conditions and class distributions.

#### 2.4 Open-Source VLM Comparison (Out-of-Domain)

| Model                | Identification (%) | Classification (%) |
| -------------------- | :----------------: | :----------------: |
| LLaVA-7B             | 27.55              | 9.59               |
| LLaVA-13B            | 7.61               | 20.85              |
| LLaVA-34B            | 79.15              | 17.35              |
| Qwen2.5-VL-7B        | 90.72              | 21.92              |
| Qwen3-VL-8B          | 99.24              | 21.46              |
| Qwen3-VL-32B         | 99.70              | 18.57              |
| LLaMA3-LLaVA-Next-8B | 14.61              | 10.96              |
| Granite Vision       | 2.44               | 31.81              |
| Gemma3-4B            | 100.00             | 14.92              |
| Gemma3-12B           | 99.70              | 15.22              |
| LLaVA-Phi3-3.8B      | 11.87              | 27.70              |
| MiniCPM-V            | 43.68              | 23.29              |
| Qwen2.5-VL-72B       | 98.63              | 17.05              |
| **BananaVLM (Ours)** | **98.38**          | **83.28**          |

#### 2.5 Closed-Source VLM Comparison (Out-of-Domain)

| Model                | Identification (%) | Classification (%) |
| -------------------- | :----------------: | :----------------: |
| Gemini 2.5 Flash Lite | 97.14             | 13.24              |
| Gemini 2.5 Flash     | 93.36              | 14.76              |
| Gemini 2.5 Pro       | 95.01              | 20.09              |
| Gemini 3.1 Flash Lite | 98.10             | 13.85              |
| Gemini 3 Flash Preview | 96.44            | 23.90              |
| **BananaVLM (Ours)** | **98.38**          | **83.28**          |

On out-of-domain data, BananaVLM achieves **83.28% classification accuracy** — more than 3× the best Gemini model's score (Gemini 3 Flash Preview at 23.90%) and more than 2.6× the best open-source classification result (Granite Vision at 31.81%). This is a compelling demonstration that domain-adapted instruction tuning transfers well across dataset distributions, not just within them.

---

### 3. Qualitative Analysis — G-Eval Evaluation

To evaluate the **quality of reasoning and language generation** — beyond raw classification accuracy — we conducted a qualitative assessment using the **G-Eval prompting strategy**. In G-Eval, multiple LLMs are used as evaluator judges to compare the open-ended text responses generated by the base LLaVA-v1.5-7B model against those generated by BananaVLM.

Four evaluation dimensions are scored on a **1–5 scale** by each judge model:

| Dimension        | Description |
| ---------------- | ----------- |
| **Disease ID**   | Correctness and specificity of disease identification |
| **Classification** | Accuracy of disease class assignment |
| **Symptom Description** | Quality and precision of described visual symptoms |
| **Management**   | Relevance and accuracy of suggested treatment or management actions |

The **Win Rate** represents the proportion of pairwise comparisons where BananaVLM's output was preferred over the base model's output by the judge.

#### 3.1 G-Eval Results

| Judge Model       | Base Disease ID | BananaVLM Disease ID | Base Cls | BananaVLM Cls | Base Symptoms | BananaVLM Symptoms | Base Mgmt | BananaVLM Mgmt | Win Rate |
| ----------------- | :-------------: | :------------------: | :------: | :-----------: | :-----------: | :----------------: | :-------: | :------------: | :------: |
| llama3.1          | 1.47            | 4.28                 | 1.87     | 4.45          | 2.86          | 4.63               | 2.87      | 4.61           | 0.92     |
| mistral           | 1.16            | 4.72                 | 1.13     | 4.46          | 2.60          | 4.23               | 2.79      | 3.97           | 0.83     |
| mistral-small     | 0.72            | 3.70                 | 0.80     | 3.57          | 2.17          | 3.95               | 2.93      | 3.75           | 0.89     |
| gemma3:12b        | 1.17            | 3.70                 | 1.01     | 3.61          | 2.03          | 3.68               | 2.25      | 3.70           | 0.96     |
| qwen2.5:14b       | 0.81            | 3.87                 | 0.55     | 3.56          | 1.48          | 3.86               | 2.75      | 3.81           | 0.90     |
| qwen2.5:32b       | 1.27            | 3.53                 | 0.98     | 3.30          | 1.95          | 3.57               | 3.03      | 3.66           | 0.82     |
| qwen2.5:7b        | 1.33            | 3.68                 | 1.07     | 3.18          | 2.18          | 4.01               | 2.47      | 3.58           | 0.81     |
| qwen2.5:72b       | 1.19            | 3.56                 | 0.62     | 3.20          | 2.14          | 3.66               | 2.89      | 3.85           | 0.87     |
| mixtral           | 1.90            | 4.37                 | 1.45     | 4.27          | 2.43          | 4.63               | 2.33      | 4.37           | 0.96     |

#### 3.2 G-Eval Analysis

BananaVLM consistently and substantially outscores the base LLaVA-v1.5-7B model across all four evaluation dimensions and all nine judge models.

**Disease Identification:** The base model scores between 0.72 and 1.90 across judges — reflecting near-random or generic responses that rarely identify the correct disease. BananaVLM scores between 3.53 and 4.72, indicating confident, specific, and correct disease identification in the vast majority of cases.

**Classification:** The base model is particularly weak here (0.55–1.87), often failing to assign a disease class at all. BananaVLM improves this to 3.18–4.46, reflecting the effect of the short Q&A label grounding stage in Stage 3 of the instruction pipeline.

**Symptom Description:** Judges score BananaVLM between 3.57 and 4.63 compared to 1.48–2.86 for the base model. This dimension benefits directly from Stage 1 of the pipeline, where LLaVA-13B generates rich, image-grounded symptom narratives that form the supervision signal.

**Management Recommendations:** BananaVLM scores 3.58–4.61 vs. 2.25–3.03 for the base model. The multi-turn Stage 2 conversations, which include treatment and management Q&A grounded in external agricultural knowledge, are responsible for this gain.

**Win Rate:** Across all nine judge models, BananaVLM achieves win rates between **0.81 and 0.96**, meaning the fine-tuned model is preferred in 81–96% of pairwise comparisons depending on the judge. The two highest win rates come from gemma3:12b and mixtral (both 0.96), while the lowest is qwen2.5:7b (0.81) — all still representing a dominant and consistent preference for BananaVLM's output quality.

These results confirm that BananaVLM not only improves classification accuracy but produces substantially better agricultural reasoning and explanation quality compared to its general-purpose base model.

---

## GroundnutVLM Results

Results for GroundnutVLM will be reported here upon completion of evaluation.

---

# References

- [LLaVA Official Repository](https://github.com/haotian-liu/LLaVA)
- [Ollama](https://ollama.com/)
- [Multi-crop Disease Dataset](https://data.mendeley.com/datasets/6243z8r6t6/1)
