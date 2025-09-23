import json
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------
# CONFIG
# ------------------------------
# - just a string: "facebook/opt-350m"
# - or a dict: {"name": "...", "subfolder": "..."}
MODELS = [
    {"name": "repelloai/MultilingualSafety_Research", "subfolder": "xlm-roberta-large-v1"},
    "Ayush-Singh/mmBert_final_model_multilingual_safety_200k",
    "Ayush-Singh/mmBert_final_model_multilingual_safety_600k"
]

REPORT_FILE = "safety_eval_report.json"


# ------------------------------
# HELPERS
# ------------------------------
def build_pipeline(model_entry):
    """Create a pipeline given either a string or dict with subfolder."""
    if isinstance(model_entry, str):
        return pipeline(
            "text-classification",
            model=model_entry,
            tokenizer=model_entry,
            truncation=True,
        )
    elif isinstance(model_entry, dict):
        model_name = model_entry["name"]
        subfolder = model_entry.get("subfolder")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, subfolder=subfolder)
        tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder, trust_remote_code=True)
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
        )
    else:
        raise ValueError(f"Unsupported model entry: {model_entry}")


def run_model_inference(model_entry, texts, batch_size=16):
    """Run classifier pipeline (binary: safe vs unsafe)."""
    clf = build_pipeline(model_entry)
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = clf(batch)
        # assume label 'LABEL_1' = unsafe, 'LABEL_0' = safe (may need to adapt per model)
        for out in outputs:
            label = out["label"]
            preds.append(1 if "1" in label or "unsafe" in label.lower() else 0)
    return preds


def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# ------------------------------
# DATASET EVALUATIONS
# ------------------------------

def eval_damo_multijail(model_entry):
    ds = load_dataset("ToxicityPrompts/DAMO-MultiJail", split="test")
    texts = ds["text"]
    langs = ds["language"]
    preds = run_model_inference(model_entry, texts)
    y_true = [1] * len(texts)
    correct = np.array(preds) == 1
    results = {"overall_accuracy": float(correct.mean()), "per_language": {}}
    for lang in set(langs):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        results["per_language"][lang] = float(correct[idxs].mean())
    return results


def eval_csrt(model_entry):
    ds = load_dataset("ToxicityPrompts/CSRT", split="code_switch")
    texts = ds["text"]
    preds = run_model_inference(model_entry, texts)
    y_true = [1] * len(texts)
    correct = np.array(preds) == 1
    return {"accuracy": float(correct.mean())}


def eval_rtplx(model_entry):
    ds = load_dataset("ToxicityPrompts/RTP-LX", split="test")
    texts = ds["Prompt"]
    langs = ds["Locale"]
    y_true = [1 if ann["Toxicity"] > 1 else 0 for ann in ds["PromptAnnotations"]]
    preds = run_model_inference(model_entry, texts)

    overall = compute_metrics(y_true, preds, average="macro")
    results = {"overall": overall, "per_locale": {}}

    for lang in set(langs):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        if not idxs:
            continue
        y_sub = [y_true[i] for i in idxs]
        p_sub = [preds[i] for i in idxs]
        results["per_locale"][lang] = compute_metrics(y_sub, p_sub, average="macro")
    return results


def eval_xsafety(model_entry):
    ds = load_dataset("ToxicityPrompts/XSafety", split="test")
    texts = ds["text"]
    cats = ds["category"]
    preds = run_model_inference(model_entry, texts)
    y_true = [1] * len(texts)
    correct = np.array(preds) == 1
    results = {"overall_accuracy": float(correct.mean()), "per_category": {}}
    for cat in set(cats):
        idxs = [i for i, c in enumerate(cats) if c == cat]
        results["per_category"][cat] = float(correct[idxs].mean())
    return results


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    final_report = {}
    for model_entry in MODELS:
        model_name = model_entry if isinstance(model_entry, str) else model_entry["name"]
        print(f"Evaluating model: {model_name}")
        model_report = {}
        model_report["DAMO-MultiJail"] = eval_damo_multijail(model_entry)
        model_report["CSRT"] = eval_csrt(model_entry)
        model_report["RTP-LX"] = eval_rtplx(model_entry)
        model_report["XSafety"] = eval_xsafety(model_entry)
        final_report[model_name] = model_report

    with open(REPORT_FILE, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n===== FINAL REPORT =====")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
