import pandas as pd
import json
import os
import requests
import time

from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_MODELS = [
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "mixtral:latest",
    "mistral:latest",
    "mistral-small:latest",
    "llama3.1:latest",
    "gemma3:12b"
]

CSV_PATH = "groundnut_report.csv"
GT_DIR = "gt_groundnut"

# save progress continuously
SAVE_EVERY = 5

# retry malformed outputs
MAX_RETRIES = 3

# =====================================================
# LOAD GROUND TRUTH ATTRIBUTES
# =====================================================

def load_gt():

    attr_dict = {}

    for file in os.listdir(GT_DIR):

        if file.endswith(".txt"):

            key = file.replace(".txt", "").lower()

            with open(
                os.path.join(GT_DIR, file),
                encoding="utf-8"
            ) as f:

                attr_dict[key] = f.read()

    return attr_dict


# =====================================================
# CALL OLLAMA
# =====================================================

def call_ollama(prompt, model_name):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        data = response.json()

        # DEBUG
        if "response" not in data:
            print("\n⚠ RAW RESPONSE:")
            print(data)

        return data.get("response", "")

    except Exception as e:

        print(f"\n❌ Request failed ({model_name}): {e}")

        return ""


# =====================================================
# PROMPT
# =====================================================

def build_prompt(gt, GT, base_out, groundnut_out):

    return f"""
You are an expert evaluator for plant disease diagnosis models.

GROUND TRUTH:
{gt}

DISEASE ATTRIBUTES:
{GT}

MODEL A (Base Model):
{base_out}

MODEL B (GroundnutVLM):
{groundnut_out}

Evaluate BOTH models on a scale of 0–5:

1. disease_identification
2. classification_accuracy
3. visible_symptoms
4. management_strategy

Be strict.
Use full scale (0–5).

Return ONLY VALID JSON.

{{
  "model_a": {{
    "disease_identification": int,
    "classification_accuracy": int,
    "visible_symptoms": int,
    "management_strategy": int
  }},
  "model_b": {{
    "disease_identification": int,
    "classification_accuracy": int,
    "visible_symptoms": int,
    "management_strategy": int
  }},
  "winner": {{
    "overall": "A/B"
  }}
}}
"""


# =====================================================
# SAFE JSON PARSE
# =====================================================

def safe_parse(text):

    try:

        start = text.find("{")

        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            return None

        cleaned = text[start:end]

        return json.loads(cleaned)

    except Exception as e:

        print("\n⚠ JSON PARSE FAILED")
        print(text)

        return None


# =====================================================
# EVALUATION
# =====================================================

def evaluate(model_name):

    df = pd.read_csv(CSV_PATH)

    attr_dict = load_gt()

    results = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        gt = str(row["ground_truth"]).lower()

        key = gt.replace(" ", "_")

        attributes = attr_dict.get(key, "")

        prompt = build_prompt(
            gt,
            attributes,
            row["base_model_output"],
            row["groundnutvlm_output"]
        )

        parsed = None

        # =================================================
        # RETRY LOOP
        # =================================================

        for attempt in range(MAX_RETRIES):

            response = call_ollama(prompt, model_name)

            parsed = safe_parse(response)

            if parsed is not None:
                break

            print(
                f"\n⚠ Retry {attempt+1}/{MAX_RETRIES}"
            )

            time.sleep(2)

        # =================================================
        # STILL FAILED
        # =================================================

        if parsed is None:

            print(
                f"\n❌ Skipping malformed output "
                f"({model_name})"
            )

            continue

        # =================================================
        # SAFE KEY ACCESS
        # =================================================

        res = {

            "base_disease":
                parsed.get("model_a", {}).get(
                    "disease_identification", 0
                ),

            "base_classification":
                parsed.get("model_a", {}).get(
                    "classification_accuracy", 0
                ),

            "base_symptoms":
                parsed.get("model_a", {}).get(
                    "visible_symptoms", 0
                ),

            "base_management":
                parsed.get("model_a", {}).get(
                    "management_strategy", 0
                ),

            "groundnut_disease":
                parsed.get("model_b", {}).get(
                    "disease_identification", 0
                ),

            "groundnut_classification":
                parsed.get("model_b", {}).get(
                    "classification_accuracy", 0
                ),

            "groundnut_symptoms":
                parsed.get("model_b", {}).get(
                    "visible_symptoms", 0
                ),

            "groundnut_management":
                parsed.get("model_b", {}).get(
                    "management_strategy", 0
                ),

            "win_overall":
                parsed.get("winner", {}).get(
                    "overall", "Unknown"
                )
        }

        results.append(res)

        # =================================================
        # CONTINUOUS SAVE
        # =================================================

        if len(results) % SAVE_EVERY == 0:

            pd.DataFrame(results).to_csv(
                f"temp_{model_name.replace(':','_')}.csv",
                index=False
            )

    return pd.DataFrame(results)


# =====================================================
# SUMMARY
# =====================================================

def summarize(df):

    metrics = [
        "disease",
        "classification",
        "symptoms",
        "management"
    ]

    summary = {}

    for m in metrics:

        summary[f"base_{m}"] = df[
            f"base_{m}"
        ].mean()

        summary[f"groundnut_{m}"] = df[
            f"groundnut_{m}"
        ].mean()

    summary["groundnut_win_rate"] = (
        df["win_overall"] == "B"
    ).mean()

    return summary


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    all_results = []

    for model in JUDGE_MODELS:

        print(f"\nRunning evaluation with {model}")

        try:

            result_df = evaluate(model)

            summary = summarize(result_df)

            summary["model"] = model

            all_results.append(summary)

        except Exception as e:

            print(
                f"\n❌ FAILED MODEL: {model}"
            )

            print(e)

            continue

    final_df = pd.DataFrame(all_results)

    final_df.to_csv(
        "multi_llm_comparison_groundnut.csv",
        index=False
    )

    # =================================================
    # SAVE TXT REPORT
    # =================================================

    with open(
        "final_llm_results_groundnut.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("MODEL COMPARISON TABLE\n")
        f.write("=" * 80 + "\n\n")

        for _, row in final_df.iterrows():

            f.write(f"Model: {row['model']}\n")

            f.write(
                f"  Base Disease: "
                f"{row['base_disease']:.2f}\n"
            )

            f.write(
                f"  Groundnut Disease: "
                f"{row['groundnut_disease']:.2f}\n"
            )

            f.write(
                f"  Base Classification: "
                f"{row['base_classification']:.2f}\n"
            )

            f.write(
                f"  Groundnut Classification: "
                f"{row['groundnut_classification']:.2f}\n"
            )

            f.write(
                f"  Base Symptoms: "
                f"{row['base_symptoms']:.2f}\n"
            )

            f.write(
                f"  Groundnut Symptoms: "
                f"{row['groundnut_symptoms']:.2f}\n"
            )

            f.write(
                f"  Base Management: "
                f"{row['base_management']:.2f}\n"
            )

            f.write(
                f"  Groundnut Management: "
                f"{row['groundnut_management']:.2f}\n"
            )

            f.write(
                f"  Groundnut Win Rate: "
                f"{row['groundnut_win_rate']:.2f}\n"
            )

            f.write("-" * 80 + "\n")

    print("\n✅ Saved:")
    print(" - multi_llm_comparison_groundnut.csv")
    print(" - final_llm_results_groundnut.txt")