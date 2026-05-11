import os
import json
import subprocess
import time
from pathlib import Path
from datetime import timedelta


DATASET_DIR=Path("dataset")
ATTRIBUTES_DIR=Path("other/attribute")
EXTERNAL_DIR=Path("other/new_external")
TEMP_DESCRIPTIONS_FILE="desc_new.jsonl"
STAGE2_FILE="banana_stage2_new.jsonl"
FINAL_OUTPUT_FILE="banana_disease_new.jsonl"
OLLAMA_TIMEOUT = 300



def read_txt_file(file_path):
    path=Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        pass
    try:
        return path.read_text(encoding="latin-1").strip()
    except UnicodeDecodeError:
        pass
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="ignore").strip()
    except Exception as e:
        print(f"Failed to read file: {file_path} | {e}")
        return ""


def call_ollama(model:str,prompt:str,image_path:str = None):
    try:
        if image_path:
            result=subprocess.run(
                ["ollama","run",model],
                input=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "images": [str(Path(image_path).resolve())],
                    "stream": False
                }).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=OLLAMA_TIMEOUT
            )
        else:
            result=subprocess.run(
                ["ollama","run",model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=OLLAMA_TIMEOUT
            )

        return result.stdout.decode(errors="ignore").strip()

    except subprocess.TimeoutExpired:
        print(f"Timeout from {model}")
        return ""
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


# STAGE 1 — IMAGE DESCRIPTIONS
def generate_descriptions(max_retries=3):
    processed = set()
    if Path(TEMP_DESCRIPTIONS_FILE).exists():
        with open(TEMP_DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(json.loads(line)["image_path"])
        print(f"Stage 1 resume: {len(processed)} images")

    all_images = [
        img for crop in DATASET_DIR.iterdir() if crop.is_dir()
        for disease in crop.iterdir() if disease.is_dir()
        for img in disease.glob("*.jpg")
    ]
    total=len(all_images)
    print(f"Found {total} images")
    start=time.time()
    done=len(processed)
    with open(TEMP_DESCRIPTIONS_FILE,"a",encoding="utf-8") as out:
        for img_path in all_images:
            rel_path = img_path.relative_to(DATASET_DIR).as_posix()
            if rel_path in processed:
                continue
            crop=img_path.parents[1].name
            disease=img_path.parent.name
            attr_file=ATTRIBUTES_DIR/crop/f"{disease}.txt"
            attributes = read_txt_file(attr_file)
            description = ""
            for _ in range(max_retries):
                prompt=(
                    f"You are an agricultural assistant. "
                    f"Describe visible symptoms in this image of a {crop} plant "
                    f"affected by {disease}. Focus only on what is visible."
                )
                description=call_ollama("llava:13b",prompt,img_path)
                if description:
                    break
                time.sleep(2)

            record={
                "image_path": rel_path,
                "crop": crop,
                "disease": disease,
                "attributes": attributes,
                "description": description if description else "[FAILED]"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            elapsed=time.time()-start
            avg=elapsed/max(1,done-len(processed))
            eta=avg*(total-done)
            print(
                f"[Stage 1] {done}/{total} | "
                f"Elapsed {timedelta(seconds=int(elapsed))} | "
                f"ETA {timedelta(seconds=int(eta))}"
            )

    print(" Stage 1 complete")


# STAGE 2 — MULTI-TURN QA

def generate_multiturn_qa():
    processed=set()
    if Path(STAGE2_FILE).exists():
        with open(STAGE2_FILE, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(json.loads(line)["image_path"])
        print(f"Stage 2 resume: {len(processed)} images")

    with open(TEMP_DESCRIPTIONS_FILE, "r", encoding="utf-8") as infile, \
         open(STAGE2_FILE, "a", encoding="utf-8") as stage2_out:

        for line in infile:
            data = json.loads(line)
            if data["image_path"] in processed:
                continue
            class_label = data["disease"]  
            dataset_name = data["crop"]    
            ext_path = EXTERNAL_DIR / dataset_name / f"{class_label}.txt"
            external_knowledge = read_txt_file(ext_path)
            print(f"[Stage 2] {data['image_path']}")

            prompt = f"""
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

            multiturn=call_ollama("mistral", prompt)
            data["external_knowledge"] = external_knowledge
            data["multi_turn_conversation"] = multiturn if multiturn else "[FAILED]"
            stage2_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            stage2_out.flush()

    print(" Stage 2 complete")



# STAGE 3 — SIMPLE QA

def generate_simple_qa():
    processed = set()
    if Path(FINAL_OUTPUT_FILE).exists():
        with open(FINAL_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(json.loads(line)["image_path"])
        print(f"Stage 3 resume: {len(processed)} images")
    with open(STAGE2_FILE, "r", encoding="utf-8") as stage2_in, \
         open(FINAL_OUTPUT_FILE, "a", encoding="utf-8") as outfile:

        for line in stage2_in:
            data = json.loads(line)
            if data["image_path"] in processed:
                continue
            print(f"[Stage 3] {data['image_path']}")
            prompt = f"""
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

            simple_qa=call_ollama("mistral", prompt)
            data["simple_qa"] = simple_qa if simple_qa else "[FAILED]"
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            outfile.flush()

    print(" Stage 3 complete")

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


if __name__=="__main__":

    total_start=time.time()

    print("\n Stage 1: Image Descriptions")
    stage1_start=time.time()
    generate_descriptions()
    stage1_time=time.time()-stage1_start
    print(f"Stage 1 completed in {format_time(stage1_time)}")

    print("\nStage 2:Multi-turn QA")
    stage2_start=time.time()
    generate_multiturn_qa()
    stage2_time=time.time()-stage2_start
    print(f"Stage 2 completed in {format_time(stage2_time)}")

    print("\n Stage 3: Simple QA")
    stage3_start=time.time()
    generate_simple_qa()
    stage3_time=time.time()-stage3_start
    print(f"Stage 3 completed in {format_time(stage3_time)}")

    total_time=time.time()-total_start

    print("\nDataset generation complete.")
    print("Final file:",FINAL_OUTPUT_FILE)
    print(f"Total runtime: {format_time(total_time)}")