import os
import io
import sys
import re
import time
import torch
import pandas as pd
import unicodedata
from datetime import datetime
from types import SimpleNamespace
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from peft import PeftModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
if not hasattr(LlamaForCausalLM, "_patched"):
    _orig = LlamaForCausalLM.forward
    def _forward_fix(self, *args, **kwargs):
        kwargs.pop("cache_position", None)
        return _orig(self, *args, **kwargs)
    LlamaForCausalLM.forward = _forward_fix
    LlamaForCausalLM._patched = True


PROMPT = "Is this crop diseased or not? Answer only yes or no."
DATASET_DIR = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/groundnut_ood_test"
OUTPUT_DIR = "./binary_crop_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def run_single(model_path, model_base, img_path, prompt):

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_base,
        model_base=None,
        model_name=get_model_name_from_path(model_base),
        device_map="cuda"
    )
    state_dict = torch.load(
        f"{model_path}/non_lora_trainables.bin",
        map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.half().eval()
    from PIL import Image
    image = Image.open(img_path).convert("RGB")
    conv = conv_templates["llava_v1"].copy()
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt_input = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_input,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    image_tensor = process_images(
        [image], image_processor, model.config
    )[0].unsqueeze(0).to("cuda").half()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=20
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if prompt in output:
        output = output.split(prompt)[-1]

    return output.strip().split("\n")[0].strip()
def evaluate_crop(name, path, model_path, model_base=None):
    print("\nEvaluating:", name)
    print("-" * 40)

    data = []
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                data.append((os.path.join(cls_path, f), cls))

    if len(data) == 0:
        print("No data found")
        return

    results = []
    total = len(data)
    correct = 0

    tp = tn = fp = fn = 0
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i, (img_path, gt_label) in enumerate(data, 1):
        print(f"[{i}/{total}] {os.path.basename(img_path)}")

        t0 = time.time()
        pred_text = run_single(model_path, model_base, img_path, PROMPT)
        latency = time.time() - t0
        latencies.append(latency)

        gt = get_gt(gt_label)
        pred = get_pred(pred_text)

        if (gt == pred):
            correct += 1

        if gt == "disease" and pred == "disease":
            tp += 1
        elif gt == "healthy" and pred == "healthy":
            tn += 1
        elif gt == "healthy" and pred == "disease":
            fp += 1
        elif gt == "disease" and pred == "healthy":
            fn += 1

        results.append({
            "image": img_path,
            "ground_truth": gt,
            "prediction_raw": pred_text,
            "prediction": pred,
            "result": "correct" if gt == pred else "incorrect",
            "latency": latency
        })

        if i % 5 == 0:
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, f"{name}_progress.csv"),
                index=False
            )
            print("saved till", i)

    acc = correct / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

    avg_lat = sum(latencies) / len(latencies)
    thr = 1 / avg_lat if avg_lat > 0 else 0

    gpu = 0
    if torch.cuda.is_available():
        gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))
    print("\nTP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
    print("Latency:", round(avg_lat, 4), "sec")
    print("Throughput:", round(thr, 2), "img/sec")
    print("GPU:", round(gpu, 2), "MB")
    df = pd.DataFrame(results)
    df["accuracy"] = [acc] + [""] * (len(df) - 1)
    df["precision"] = [prec] + [""] * (len(df) - 1)
    df["recall"] = [rec] + [""] * (len(df) - 1)
    df["f1"] = [f1] + [""] * (len(df) - 1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{name}_final_{ts}.csv")
    df.to_csv(out_path, index=False)
    print("saved final:", out_path)



if __name__ == "__main__":

    MODEL_PATH = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/checkpoints/llava-v1.5-7b-dora-groundnut-vlm_new_epoch_7"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"
    for crop in os.listdir(DATASET_DIR):
        crop_path = os.path.join(DATASET_DIR, crop)
        if os.path.isdir(crop_path):
            evaluate_crop(crop, crop_path, MODEL_PATH, MODEL_BASE)



"""gurations of models and can yield errors.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.54s/it]
[905/905] sig_extra_0334.jpg
You are using a model of type llava to instantiate a model of type llava_llama. This is not supported for all configurations of models and can yield errors.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.47s/it]
saved till 905

Accuracy: 96.8 %
Precision: 0.9752
Recall: 0.9887
F1: 0.9819

TP: 788 TN: 88 FP: 20 FN: 9
Latency: 9.0304 sec
Throughput: 0.11 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/banana_final_20260416_041234.csv"""

"""
Accuracy: 96.13 %
Precision: 0.9669
Recall: 0.99
F1: 0.9783

TP: 789 TN: 81 FP: 27 FN: 8
Latency: 9.0616 sec
Throughput: 0.11 img/sec
GPU: 14846.82 MB"""
"""
Accuracy: 96.02 %
Precision: 0.9657
Recall: 0.99
F1: 0.9777

TP: 789 TN: 80 FP: 28 FN: 8
Latency: 9.0469 sec
Throughput: 0.11 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/banana_final_20260419_234044.csv"""


# groundnut
"""epoch 3 indomain"""
"""Accuracy: 96.35 %
Precision: 0.9887
Recall: 0.9728
F1: 0.9807

TP: 786 TN: 32 FP: 9 FN: 22
Latency: 10.4006 sec
Throughput: 0.1 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/groundnut_final_20260512_211442.csv"""

""" out domain epoch 3
Accuracy: 92.9 %
Precision: 0.9389
Recall: 0.9784
F1: 0.9582

TP: 814 TN: 115 FP: 53 FN: 18
Latency: 10.8075 sec
Throughput: 0.09 img/sec
GPU: 14846.82 MB ood"""

"""epoch 5 indomain 
Accuracy: 96.23 %
Precision: 0.9826
Recall: 0.9777
F1: 0.9801

TP: 790 TN: 27 FP: 14 FN: 18
Latency: 9.7224 sec
Throughput: 0.1 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/groundnut_final_20260514_222232.csv"""
"""epoch 5 out domain

Accuracy: 91.9 %
Precision: 0.9214
Recall: 0.9868
F1: 0.953

TP: 821 TN: 98 FP: 70 FN: 11
Latency: 9.6925 sec
Throughput: 0.1 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/groundnut_final_20260514_224822.csv"""


"""epoch 7 in domain 

Accuracy: 96.47 %
Precision: 0.9779
Recall: 0.9851
F1: 0.9815

TP: 796 TN: 23 FP: 18 FN: 12
Latency: 16.6824 sec
Throughput: 0.06 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/groundnut_final_20260516_074906.csv"""

"""Accuracy: 90.5 %
Precision: 0.9027
Recall: 0.9928
F1: 0.9456

TP: 826 TN: 79 FP: 89 FN: 6
Latency: 17.5612 sec
Throughput: 0.06 img/sec
GPU: 14846.82 MB
saved final: ./binary_crop_reports/groundnut_final_20260516_124854.csv epoch 7 out domain"""