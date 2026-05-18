"""import os
import io
import sys
import re
import time
import torch
import pandas as pd
import unicodedata
import subprocess
from datetime import datetime
from difflib import SequenceMatcher

# 🔥 NEW IMPORTS (DoRA)
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

# Patch
if not hasattr(LlamaForCausalLM,"_patched_forward"):
    original_forward=LlamaForCausalLM.forward
    def forward_wrapper(self,*args,**kwargs):
        kwargs.pop("cache_position",None)
        return original_forward(self,*args,**kwargs)
    LlamaForCausalLM.forward=forward_wrapper
    LlamaForCausalLM._patched_forward=True


PROMPT = "Identify the disease present in this crop image. Respond with only the disease name."
DATASET_DIR = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/test"
OUTPUT_DIR = "./crop_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# GPU UTIL
# =========================
def get_gpu_memory():
    try:
        result=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(result.decode("utf-8").strip().split("\n")[0])
    except:
        return -1


# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(text):
    if not text:
        return ""
    text=text.lower()
    text=unicodedata.normalize("NFKD", text)
    text=re.sub(r"\(.*?\)", "", text)
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    text=re.sub(r"\s+", " ", text)
    return text.strip()


def check_match(gt,pred,sim_thresh=0.6,overlap_thresh=0.5):
    gt_clean=normalize_text(gt)
    pred_clean=normalize_text(pred)

    if not gt_clean or not pred_clean:
        return False

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    gt_tokens = set(gt_clean.split())
    pred_tokens = set(pred_clean.split())

    if gt_tokens and pred_tokens:
        overlap=len(gt_tokens&pred_tokens)/len(gt_tokens|pred_tokens)
        if overlap>=overlap_thresh:
            return True

    similarity=SequenceMatcher(None,gt_clean,pred_clean).ratio()
    return similarity >= sim_thresh


# =========================
# 🔥 LOAD DORA MODEL ONCE
# =========================
def load_dora_model(model_path, model_base):

    print("🔄 Loading DoRA model...")

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

    return tokenizer, model, image_processor


# =========================
# 🔥 INFERENCE (NO RELOAD)
# =========================
def infer_single(tokenizer, model, image_processor, image_path, prompt):

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

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
            max_new_tokens=256
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if prompt in output:
        output = output.split(prompt)[-1]

    return output.strip().split("\n")[0].strip()


# =========================
# EVALUATION
# =========================
def evaluate_crop(crop_name,crop_path,model_path,model_base=None):

    print(f"\nEvaluating: {crop_name}")

    tokenizer, model, image_processor = load_dora_model(model_path, model_base)

    samples=[]
    for disease in sorted(os.listdir(crop_path)):
        disease_dir=os.path.join(crop_path,disease)
        if not os.path.isdir(disease_dir):
            continue
        for img in os.listdir(disease_dir):
            if img.lower().endswith((".jpg",".png",".jpeg")):
                samples.append((os.path.join(disease_dir, img),disease))

    if not samples:
        print("No valid images found.")
        return

    results=[]
    correct=0

    latencies=[]
    gpu_usage_list=[]
    start_total=time.time()

    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path=os.path.join(
        OUTPUT_DIR,
        f"{crop_name}_report_{timestamp}.csv"
    )

    for idx,(img_path,gt_label) in enumerate(samples,1):
        print(f"[{idx}/{len(samples)}] {os.path.basename(img_path)}")

        start_time=time.time()
        gpu_before=get_gpu_memory()

        prediction=infer_single(tokenizer, model, image_processor, img_path, PROMPT)

        gpu_after=get_gpu_memory()
        end_time=time.time()

        latency=end_time-start_time
        gpu_used=max(gpu_after-gpu_before,0)

        latencies.append(latency)
        gpu_usage_list.append(gpu_used)

        is_correct = check_match(gt_label, prediction)
        if is_correct:
            correct += 1

        results.append({
            "Image": img_path,
            "Ground Truth": gt_label,
            "Prediction": prediction,
            "Result": "Correct" if is_correct else "Incorrect",
            "Latency (s)": round(latency, 3),
            "GPU Used (MB)": gpu_used
        })

        # ✅ SAVE EVERY 5 (UNCHANGED LOGIC)
        if idx%5==0:
            temp_acc=(correct/idx)*100
            df_temp=pd.DataFrame(results)
            df_temp["Running Accuracy (%)"]=[temp_acc]+[""]*(len(df_temp)-1)
            df_temp.to_csv(csv_path,index=False)
            print(f"Saved checkpoint at {idx} samples")

    total_time=time.time()-start_total
    total_samples=len(samples)
    accuracy=(correct/total_samples)*100
    avg_latency=sum(latencies)/len(latencies)
    throughput=total_samples/total_time
    avg_gpu=sum(gpu_usage_list)/len(gpu_usage_list)

    print("\nFinal Results")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total_samples})")
    print(f"Avg Latency: {avg_latency:.3f} s/image")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Avg GPU Usage per inference: {avg_gpu:.2f} MB")

    df_final=pd.DataFrame(results)
    df_final["Final Accuracy (%)"]=[accuracy]+[""]*(len(df_final)-1)
    df_final.to_csv(csv_path,index=False)

    print(f"Report saved at: {csv_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    MODEL_PATH = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/checkpoints/llava-v1.5-7b-dora-banana-vlm_epoch_3"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"

    for crop in sorted(os.listdir(DATASET_DIR)):
        crop_dir=os.path.join(DATASET_DIR, crop)
        if os.path.isdir(crop_dir):
            evaluate_crop(
                crop_name=crop,
                crop_path=crop_dir,
                model_path=MODEL_PATH,
                model_base=MODEL_BASE
            )

    print("\nAll evaluations completed.\n")

"""
"""Final Results
Accuracy: 34.50% (275/797)
Avg Latency: 1.429 s/image
Throughput: 0.70 images/sec
Avg GPU Usage per inference: 0.71 MB
Report saved at: ./crop_reports/banana_report_20260416_042027.csv"""
import os
import re
import time
import torch
import pandas as pd
import unicodedata
import subprocess
from datetime import datetime
from difflib import SequenceMatcher

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

# =========================
# PATCH
# =========================
if not hasattr(LlamaForCausalLM,"_patched_forward"):
    original_forward=LlamaForCausalLM.forward
    def forward_wrapper(self,*args,**kwargs):
        kwargs.pop("cache_position",None)
        return original_forward(self,*args,**kwargs)
    LlamaForCausalLM.forward=forward_wrapper
    LlamaForCausalLM._patched_forward=True


PROMPT = "Identify the disease present in this crop image. Respond with only the disease name."
DATASET_DIR = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/groundnut_ood_test"
OUTPUT_DIR = "./crop_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# GPU MEMORY
# =========================
def get_gpu_memory():
    try:
        result=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(result.decode("utf-8").strip().split("\n")[0])
    except:
        return -1


# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(text):
    if not text:
        return ""
    text=text.lower()
    text=unicodedata.normalize("NFKD", text)
    text=re.sub(r"\(.*?\)", "", text)
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    text=re.sub(r"\s+", " ", text)
    return text.strip()


def check_match(gt,pred,sim_thresh=0.6,overlap_thresh=0.5):
    gt_clean=normalize_text(gt)
    pred_clean=normalize_text(pred)

    if not gt_clean or not pred_clean:
        return False

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    gt_tokens=set(gt_clean.split())
    pred_tokens=set(pred_clean.split())

    if gt_tokens and pred_tokens:
        overlap=len(gt_tokens&pred_tokens)/len(gt_tokens|pred_tokens)
        if overlap>=overlap_thresh:
            return True

    similarity=SequenceMatcher(None,gt_clean,pred_clean).ratio()
    return similarity>=sim_thresh


# =========================
# 🔥 LOAD DORA PER IMAGE
# =========================
def load_dora_model(model_path, model_base):

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

    return tokenizer, model, image_processor


# =========================
# 🔥 INFERENCE (LOAD EACH TIME)
# =========================
def infer_single(model_path, model_base, image_path, prompt):

    # LOAD MODEL EVERY TIME
    tokenizer, model, image_processor = load_dora_model(model_path, model_base)

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

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
            max_new_tokens=256
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if prompt in output:
        output = output.split(prompt)[-1]

    output = output.strip().split("\n")[0].strip()

    # 🔥 VERY IMPORTANT: FREE MEMORY
    del model
    del tokenizer
    del image_tensor
    torch.cuda.empty_cache()

    return output


# =========================
# EVALUATION (UNCHANGED)
# =========================
def evaluate_crop(crop_name,crop_path,model_path,model_base=None):

    print(f"\nEvaluating: {crop_name}")

    samples=[]
    for disease in sorted(os.listdir(crop_path)):
        disease_dir=os.path.join(crop_path,disease)
        if not os.path.isdir(disease_dir):
            continue
        for img in os.listdir(disease_dir):
            if img.lower().endswith((".jpg",".png",".jpeg")):
                samples.append((os.path.join(disease_dir, img),disease))

    if not samples:
        print("No valid images found.")
        return

    results=[]
    correct=0

    latencies=[]
    gpu_usage_list=[]
    start_total=time.time()

    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path=os.path.join(
        OUTPUT_DIR,
        f"{crop_name}_report_{timestamp}.csv"
    )

    for idx,(img_path,gt_label) in enumerate(samples,1):
        print(f"[{idx}/{len(samples)}] {os.path.basename(img_path)}")

        start_time=time.time()
        gpu_before=get_gpu_memory()

        prediction=infer_single(model_path,model_base,img_path,PROMPT)

        gpu_after=get_gpu_memory()
        end_time=time.time()

        latency=end_time-start_time
        gpu_used=max(gpu_after-gpu_before,0)

        latencies.append(latency)
        gpu_usage_list.append(gpu_used)

        is_correct = check_match(gt_label, prediction)
        if is_correct:
            correct += 1

        results.append({
            "Image": img_path,
            "Ground Truth": gt_label,
            "Prediction": prediction,
            "Result": "Correct" if is_correct else "Incorrect",
            "Latency (s)": round(latency, 3),
            "GPU Used (MB)": gpu_used
        })

        # ✅ SAVE EVERY 5
        if idx%5==0:
            temp_acc=(correct/idx)*100
            df_temp=pd.DataFrame(results)
            df_temp["Running Accuracy (%)"]=[temp_acc]+[""]*(len(df_temp)-1)
            df_temp.to_csv(csv_path,index=False)
            print(f"Saved checkpoint at {idx} samples")

    total_time=time.time()-start_total
    total_samples=len(samples)
    accuracy=(correct/total_samples)*100
    avg_latency=sum(latencies)/len(latencies)
    throughput=total_samples/total_time
    avg_gpu=sum(gpu_usage_list)/len(gpu_usage_list)

    print("\nFinal Results")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total_samples})")
    print(f"Avg Latency: {avg_latency:.3f} s/image")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Avg GPU Usage per inference: {avg_gpu:.2f} MB")

    df_final=pd.DataFrame(results)
    df_final["Final Accuracy (%)"]=[accuracy]+[""]*(len(df_final)-1)
    df_final.to_csv(csv_path,index=False)

    print(f"Report saved at: {csv_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    MODEL_PATH = "/home/user/Downloads/agrogpt-20260218T141948Z-1-001/agrogpt/LLaVA/checkpoints/llava-v1.5-7b-dora-groundnut-vlm_new_epoch_7"
    MODEL_BASE = "liuhaotian/llava-v1.5-7b"

    for crop in sorted(os.listdir(DATASET_DIR)):
        crop_dir=os.path.join(DATASET_DIR, crop)
        if os.path.isdir(crop_dir):
            evaluate_crop(
                crop_name=crop,
                crop_path=crop_dir,
                model_path=MODEL_PATH,
                model_base=MODEL_BASE
            )

    print("\nAll evaluations completed.\n")

    """Final Results
Accuracy: 34.50% (275/797)
Avg Latency: 9.907 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 0.98 MB
Report saved at: ./crop_reports/banana_report_20260416_044107.csv"""
"""Final Results
Accuracy: 19.45% (155/797)
Avg Latency: 9.811 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 0.95 MB
Report saved at: ./crop_reports/banana_report_20260420_031322.csv"""
"""Final Results
Accuracy: 8.83% (58/657)
Avg Latency: 9.694 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 1.17 MB
Report saved at: ./crop_reports/banana_report_20260426_235908.csv

Evaluating: ood_result
No valid images found.

All evaluations completed."""
"""Final Results
Accuracy: 8.98% (59/657)
Avg Latency: 9.883 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 0.93 MB
Report saved at: ./crop_reports/banana_report_20260427_015727.csv

Evaluating: ood_result
No valid images found.

All evaluations completed."""
""""Final Results
Accuracy: 7.15% (47/657)
Avg Latency: 9.863 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 1.05 MB
Report saved at: ./crop_reports/banana_report_20260427_034633.csv

Evaluating: ood_result
No valid images found.

All evaluations completed."""









"""
Final Results
Accuracy: 36.75% (312/849)
Avg Latency: 10.650 s/image
Throughput: 0.09 images/sec
Avg GPU Usage per inference: 148.80 MB
Report saved at: ./crop_reports/groundnut_report_20260512_213312.csv

All evaluations completed. in domain"""


"""
Final Results
Accuracy: 33.30% (333/1000)
Avg Latency: 10.484 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 94.88 MB
Report saved at: ./crop_reports/groundnut_report_20260512_213348.csv

All evaluations completed.
out domain"""

"""epoch 5 in domain 
Final Results
Accuracy: 38.61% (312/808)
Avg Latency: 9.640 s/image
Throughput: 0.10 images/sec
Avg GPU Usage per inference: 118.64 MB
Report saved at: ./crop_reports/groundnut_report_20260514_222857.csv

Evaluating: healthy
No valid images found.

All evaluations completed."""

"""epoch 5 ood
Final Results
Accuracy: 40.02% (333/832)
Avg Latency: 9.500 s/image
Throughput: 0.11 images/sec
Avg GPU Usage per inference: 32.48 MB
Report saved at: ./crop_reports/groundnut_report_20260514_234125.csv

Evaluating: healthy
No valid images found.

All evaluations completed.
"""
"""Final Results
Accuracy: 38.49% (311/808)
Avg Latency: 17.870 s/image
Throughput: 0.06 images/sec
Avg GPU Usage per inference: 0.00 MB
Report saved at: ./crop_reports/groundnut_report_20260516_183016.csv

Evaluating: healthy
No valid images found.

All evaluations completed. epoch 7 in domain"""
"""  warnings.warn(
You are using a model of type llava to instantiate a model of type llava_llama. This is not supported for all configurations of models and can yield errors.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.53s/it]
/home/user/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:392: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(

Final Results
Accuracy: 40.02% (333/832)
Avg Latency: 17.577 s/image
Throughput: 0.06 images/sec
Avg GPU Usage per inference: 0.00 MB
Report saved at: ./crop_reports/groundnut_report_20260516_224838.csv

Evaluating: healthy
No valid images found.

All evaluations completed. 7 ood"""