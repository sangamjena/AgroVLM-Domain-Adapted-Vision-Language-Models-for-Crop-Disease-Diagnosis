import os
import io
import sys
import re
import time
import torch
import pandas as pd
import unicodedata
import numpy as np
from types import SimpleNamespace
from datetime import datetime
from difflib import SequenceMatcher

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers.models.llama.modeling_llama import LlamaForCausalLM


# ================= PATCH =================
if not getattr(LlamaForCausalLM, "_cachepos_patched", False):
    _orig_forward_llama = LlamaForCausalLM.forward

    def _patched_forward_llama(self, *args, **kwargs):
        kwargs.pop("cache_position", None)
        return _orig_forward_llama(self, *args, **kwargs)

    LlamaForCausalLM.forward = _patched_forward_llama
    LlamaForCausalLM._cachepos_patched = True


# ================= CONFIG =================
DATASET_DIR = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/indom"
OUTPUT_DIR = "./crop_reports/groundnut"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= CLEAN =================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ================= ZERO-SHOT PROMPT =================
def build_prompt(class_names):
    class_str = ", ".join(class_names)
    return f"""
Choose the correct disease from:
[{class_str}]

Return ONLY one label from the list.
"""


# ================= MODEL =================
def run_eval_single(model_path, model_base, image_file, prompt):

    args = SimpleNamespace(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        query=prompt,
        image_file=image_file,

        conv_mode=None,   # ✅ required
        sep=",",          # ✅ FIX THIS ERROR

        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=50
    )

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        eval_model(args)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    lines = [ln.strip() for ln in output.split("\n") if ln.strip()]
    return lines[-1] if lines else ""

# ================= MAIN =================
def evaluate_crop(crop_name, crop_path, model_path, model_base=None):

    print(f"\n🌾 Evaluating Crop: {crop_name}")

    dataset = []
    class_names = sorted(os.listdir(crop_path))

    for cls in class_names:
        cls_path = os.path.join(crop_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for img in os.listdir(cls_path):
            if img.endswith((".jpg", ".png")):
                dataset.append((os.path.join(cls_path, img), cls))

    prompt = build_prompt(class_names)

    y_true, y_pred, latencies = [], [], []

    # 🔥 Create CSV path BEFORE loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{OUTPUT_DIR}/{crop_name}_report_{timestamp}.csv"

    total_start = time.time()

    for idx, (img_path, gt) in enumerate(dataset, 1):
        print(f"[{idx}/{len(dataset)}] {os.path.basename(img_path)}")

        start = time.time()
        pred = run_eval_single(model_path, model_base, img_path, prompt)
        latency = time.time() - start

        latencies.append(latency)
        y_true.append(gt)
        y_pred.append(pred)

        # ================= 💾 SAVE EVERY 5 IMAGES =================
        if idx % 5 == 0:
            df_partial = pd.DataFrame({
                "GT": y_true,
                "Prediction": y_pred,
                "Latency": latencies
            })
            df_partial.to_csv(csv_path, index=False)
            print(f"💾 Saved checkpoint at {idx} images → {csv_path}")

    total_time = time.time() - total_start

    # ================= METRICS =================
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("\n🧾 Confusion Matrix:\n", cm)

    # ================= ROC AUC =================
    try:
        y_true_bin = label_binarize(y_true, classes=class_names)
        y_pred_bin = label_binarize(y_pred, classes=class_names)

        auc_score = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
        print(f"\n📈 AUC Score: {auc_score:.4f}")
    except:
        print("\n⚠️ AUC could not be computed")

    # ================= SYSTEM =================
    avg_latency = sum(latencies) / len(latencies)
    throughput = len(dataset) / total_time

    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n⚡ System Metrics:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} img/s")
    print(f"GPU Memory: {gpu_mem:.2f} MB")

    # ================= FINAL SAVE =================
    df = pd.DataFrame({
        "GT": y_true,
        "Prediction": y_pred,
        "Latency": latencies
    })

    df.to_csv(csv_path, index=False)
    print(f"📁 Final report saved: {csv_path}")


# ================= RUN =================
if __name__ == "__main__":

    MODEL_PATH = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/checkpoints/llava-v1.5-7b-lora-groundnut-vlm_new_epoch_7"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"

    for crop in os.listdir(DATASET_DIR):
        crop_path = os.path.join(DATASET_DIR, crop)
        if os.path.isdir(crop_path):
            evaluate_crop(crop, crop_path, MODEL_PATH, MODEL_BASE)

    print("\n🎯 Done!")