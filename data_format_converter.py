import json
import uuid
import re
from pathlib import Path

INPUT_FILE = "banana_disease_new.jsonl"
OUTPUT_FILE = "banana.json"      

def parse_qa_block(text):
   
    if not text or not isinstance(text, str):
        return []
    qa_pairs = []
    current_q = None
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("q"):
            q = re.sub(
                r"^q[\d/]*\s*[:\-]?\s*",
                "",
                line,
                flags=re.I
            ).strip()
            if q:
                current_q = q
        elif lower.startswith("a") or lower.startswith("answer"):
            a = re.sub(
                r"^answer\s*\d*\s*(\(a\d*\))?\s*[:\-]?\s*",
                "",
                line,
                flags=re.I
            )
            a = re.sub(
                r"^a\d*\s*[:\-]?\s*",
                "",
                a,
                flags=re.I
            ).strip()
            if current_q and a:
                qa_pairs.append((current_q, a))
                current_q = None

    return qa_pairs



def convert_to_llava_format():
    output_records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            conversations = []
            if record.get("description"):
                conversations.append({
                    "from": "human",
                    "value": "<image>\nDescribe the image."
                })
                conversations.append({
                    "from": "gpt",
                    "value": record["description"].strip()
                })
            if record.get("multi_turn_conversation"):
                qa_pairs = parse_qa_block(record["multi_turn_conversation"])
                for q, a in qa_pairs:
                    conversations.append({
                        "from": "human",
                        "value": q
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": a
                    })
            if record.get("simple_qa"):
                qa_pairs = parse_qa_block(record["simple_qa"])
                for q, a in qa_pairs:
                    conversations.append({
                        "from": "human",
                        "value": q
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": a
                    })
            if not conversations:
                continue
            output_records.append({
                "id": str(uuid.uuid4()),
                "image": record["image_path"],
                "conversations": conversations
            })
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(output_records)} records into LLaVA format")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    convert_to_llava_format()
