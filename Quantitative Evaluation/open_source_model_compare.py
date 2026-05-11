

import os
import base64
import requests
import csv
from pathlib import Path
from tqdm import tqdm
import unicodedata
import re
from difflib import SequenceMatcher

DATASET_DIR = Path("/home/user/Arun/Agri_Vision/indom")

PROMPT_bi = "Tell whether the plant shown is healthy or not. Answer strictly yes if the plant is diseased, and no if it is healthy."
PROMPT_bc = "Identify the disease present in this crop image. Respond with only the disease name."

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_API_URL = "http://localhost:11434/api/generate"


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(model, image_path, prompt):
    img_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"(error:{e})"


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_match(gt, pred, sim_thresh=0.6, overlap_thresh=0.5):
    gt_clean = normalize_text(gt)
    pred_clean = normalize_text(pred)

    if not gt_clean or not pred_clean:
        return False

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    gt_tokens = set(gt_clean.split())
    pred_tokens = set(pred_clean.split())

    if gt_tokens and pred_tokens:
        overlap = len(gt_tokens & pred_tokens) / len(gt_tokens | pred_tokens)
        if overlap >= overlap_thresh:
            return True

    similarity = SequenceMatcher(None, gt_clean, pred_clean).ratio()
    return similarity >= sim_thresh


def clean_text(x):
    if x is None:
        return ""
    x = str(x).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r'[^a-z0-9\s]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()


def get_gt(label):
    label = clean_text(label)
    if "healthy" in label:
        return "healthy"
    return "disease"


def get_pred(pred):
    pred = clean_text(pred)
    if "yes" in pred:
        return "disease"
    if "no" in pred:
        return "healthy"
    return "unknown"


def disease_identification_eval(model, dataset, task):

    disease_folders = [
        f for f in DATASET_DIR.iterdir()
        if f.is_dir()
    ]

    if task == "id":
        PROMPT = PROMPT_bi
    elif task == "cls":
        PROMPT = PROMPT_bc
    else:
        raise ValueError("Invalid task")

    rows = []

    for disease_dir in disease_folders:
        disease = disease_dir.name

        images = [
            p for p in disease_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".png", ".jpeg")
        ]

        if not images:
            continue

        print(f"\nEvaluating: {disease}({len(images)} samples)\n")

        for img_path in tqdm(images, desc=disease, ncols=100):
            response = query_ollama(model, str(img_path), PROMPT)

            if task == "cls":
                is_correct = check_match(disease, response)
            else:
                gt_label = get_gt(disease)
                pred_label = get_pred(response)
                is_correct = (gt_label == pred_label)

            rows.append({
                "image": str(img_path),
                "true_label": disease,
                "model_response": response,
                "is_correct": is_correct
            })

    safe_model_name = (
    model
    .replace(":", "_")
    .replace("/", "_")
)

    csv_path = RESULTS_DIR / f"{safe_model_name}_{task}_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "true_label", "model_response", "is_correct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])

    print(f"\n Saved results to {csv_path}")
    print(f"Accuracy: {correct}/{total}={correct/total:.4f}")

    with open("accuracy.txt", "a") as f:
        f.write(f"{model} task={task} Accuracy: {correct}/{total}={correct/total:.4f}\n")


if __name__ == "__main__":

    models = [ "mapler/llama3-llava-next-8b"]

    datasets = ["b"]
    tasks = ["id", "cls"]

    for model in models:
        for dataset in datasets:
            for task in tasks:
                print("\n", "=" * 50)
                print("Running:", model, dataset, task)
                print("=" * 50)
                disease_identification_eval(model, dataset, task)

