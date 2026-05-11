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
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers.models.llama.modeling_llama import LlamaForCausalLM



if not hasattr(LlamaForCausalLM, "_patched"):
    _orig = LlamaForCausalLM.forward
    def _forward_fix(self, *args, **kwargs):
        if "cache_position" in kwargs:
            kwargs.pop("cache_position")
        return _orig(self, *args, **kwargs)
    LlamaForCausalLM.forward=_forward_fix
    LlamaForCausalLM._patched=True



PROMPT="Is this crop diseased or not? Answer only yes or no."
DATASET_DIR="/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/groundnut_ood_test"
OUTPUT_DIR="./binary_crop_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)



def clean_text(x):
    if x is None:
        return ""
    x=str(x).lower()
    x=unicodedata.normalize("NFKD",x)
    x=re.sub(r'[^a-z0-9\s]', ' ',x)
    x=re.sub(r'\s+', ' ', x)
    return x.strip()


def get_gt(label):
    label=clean_text(label)
    if "healthy" in label:
        return "healthy"
    return "disease"


def get_pred(pred):
    pred=clean_text(pred)
    if "yes" in pred:
        return "disease"
    if "no" in pred:
        return "healthy"
    return "unknown"


def run_single(model_path,model_base,img_path,prompt):

    args=SimpleNamespace(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        query=prompt,
        conv_mode=None,
        image_file=img_path,
        sep=",",
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=20
    )

    backup=sys.stdout
    sys.stdout=io.StringIO()

    try:
        eval_model(args)
        out=sys.stdout.getvalue()
    finally:
        sys.stdout=backup

    lines=[l.strip() for l in out.split("\n") if l.strip()]
    return lines[-1] if lines else ""



def evaluate_crop(name,path,model_path,model_base=None):
    print("\nEvaluating:",name)
    print("-" * 40)
    data=[]
    for cls in os.listdir(path):
        cls_path=os.path.join(path, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                data.append((os.path.join(cls_path, f), cls))

    if len(data)==0:
        print("No data found")
        return

    results=[]
    total=len(data)
    correct=0

    tp=tn=fp=fn=0
    latencies=[]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i, (img_path, gt_label) in enumerate(data, 1):
        print(f"[{i}/{total}] {os.path.basename(img_path)}")
        t0 = time.time()
        pred_text = run_single(model_path, model_base, img_path, PROMPT)
        latency = time.time() - t0
        latencies.append(latency)
        gt=get_gt(gt_label)
        pred=get_pred(pred_text)
        if (gt == "disease" and pred == "disease") or (gt == "healthy" and pred == "healthy"):
            is_correct = True
            correct += 1
        else:
            is_correct = False
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
            "result": "correct" if is_correct else "incorrect",
            "latency": latency
        })
        if i%5==0:
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR,f"{name}_progress.csv"),
                index=False
            )
            print("saved till",i)

  
    acc=correct/total
    prec=tp/(tp+fp) if (tp+fp)>0 else 0
    rec=tp/(tp+fn) if (tp+fn)>0 else 0
    f1=(2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0
    avg_lat=sum(latencies)/len(latencies)
    thr=1/avg_lat if avg_lat > 0 else 0
    gpu=0
    if torch.cuda.is_available():
        gpu=torch.cuda.max_memory_allocated()/(1024**2)

    print("\nAccuracy:",round(acc*100,2),"%")
    print("Precision:",round(prec,4))
    print("Recall:",round(rec,4))
    print("F1:",round(f1,4))
    print("\nTP:",tp,"TN:",tn,"FP:",fp,"FN:",fn)
    print("Latency:",round(avg_lat,4),"sec")
    print("Throughput:",round(thr,2),"img/sec")
    print("GPU:",round(gpu,2),"MB")

    df = pd.DataFrame(results)
    df["accuracy"]=[acc]+[""]*(len(df)-1)
    df["precision"]=[prec]+[""]*(len(df)-1)
    df["recall"]=[rec]+[""]*(len(df)-1)
    df["f1"]=[f1]+[""]*(len(df)-1)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path=os.path.join(OUTPUT_DIR,f"{name}_final_{ts}.csv")
    df.to_csv(out_path,index=False)
    print("saved final:",out_path)



if __name__ =="__main__":

    MODEL_PATH="/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/checkpoints/llava-v1.5-7b-lora-groundnut-vlm_new_epoch_7"
    MODEL_BASE="liuhaotian/llava-v1.5-7b"
    for crop in os.listdir(DATASET_DIR):
        crop_path=os.path.join(DATASET_DIR,crop)
        if os.path.isdir(crop_path):
            evaluate_crop(crop, crop_path, MODEL_PATH, MODEL_BASE)

    


"""TP: 654 TN: 77 FP: 9 FN: 3
Latency: 10.7735 sec
Throughput: 0.09 img/sec"""
"""Accuracy: 97.24 %
Precision: 0.9696
Recall: 1.0
F1: 0.9846

TP: 797 TN: 83 FP: 25 FN: 0
Latency: 9.2346 sec
Throughput: 0.11 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/banana_final_20260412_184100.csv

done"""
"""
Accuracy: 97.68 %
Precision: 0.9743
Recall: 1.0
F1: 0.987

TP: 797 TN: 87 FP: 21 FN: 0
Latency: 10.6664 sec
Throughput: 0.09 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/banana_final_20260412_212338.csv"""
"""Accuracy: 99.6 %
Precision: 0.9955
Recall: 1.0
F1: 0.9977

TP: 657 TN: 83 FP: 3 FN: 0
Latency: 9.3167 sec
Throughput: 0.11 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/banana_final_20260420_072332.csv

Evaluating: ood_result
---------------------------------------- epoch 5"""
"""
Accuracy: 99.73 %
Precision: 0.997
Recall: 1.0
F1: 0.9985

TP: 657 TN: 84 FP: 2 FN: 0
Latency: 9.664 sec
Throughput: 0.1 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/banana_final_20260420_092619.csv epoch 3"""







"""Accuracy: 96.7 %
Precision: 0.9756
Recall: 0.9901
F1: 0.9828

TP: 800 TN: 21 FP: 20 FN: 8
Latency: 10.1734 sec
Throughput: 0.1 img/sec
GPU: 23637.22 MB
saved final: ./binary_crop_reports/groundnut_final_20260510_064630.csv"""
"""Accuracy: 98.94 %
Precision: 0.9938
Recall: 0.995
F1: 0.9944

TP: 804 TN: 36 FP: 5 FN: 4
Latency: 10.8352 sec
Throughput: 0.09 img/sec
GPU: 14843.24 MB"""

"""
Accuracy: 99.18 %
Precision: 0.9938
Recall: 0.9975
F1: 0.9957

TP: 806 TN: 36 FP: 5 FN: 2
Latency: 10.8306 sec
Throughput: 0.09 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/groundnut_final_20260510_092156.csv"""



""""""
"""
Accuracy: 97.1 %
Precision: 0.9663
Recall: 1.0
F1: 0.9829

TP: 832 TN: 139 FP: 29 FN: 0
Latency: 9.121 sec
Throughput: 0.11 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/groundnut_final_20260510_172324.csv"""


"""Accuracy: 97.0 %
Precision: 0.9652
Recall: 1.0
F1: 0.9823

TP: 832 TN: 138 FP: 30 FN: 0
Latency: 10.5712 sec
Throughput: 0.09 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/groundnut_final_20260510_144803.csv"""
"""Accuracy: 96.9 %
Precision: 0.9641
Recall: 1.0
F1: 0.9817

TP: 832 TN: 137 FP: 31 FN: 0
Latency: 10.5496 sec
Throughput: 0.09 img/sec
GPU: 14843.24 MB
saved final: ./binary_crop_reports/groundnut_final_20260510_144753.csv"""