import json
import gc
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
MODELS = [
    {
        "name": "repelloai/MultilingualSafety_Research",
        "subfolder": "xlm-roberta-large-v1",
        "label_map": {"SAFE": 0, "UNSAFE": 1},
    },
    {
        "name": "Ayush-Singh/mmBert_final_model_multilingual_safety_200k",
        "label_map": {"LABEL_0": 0, "LABEL_1": 1},
    },
    {
        "name": "Ayush-Singh/mmBert_final_model_multilingual_safety_600k",
        "label_map": {"LABEL_0": 0, "LABEL_1": 1},
    },
]

REPORT_FILE = "safety_eval_report.json"

# ------------------------------
# HELPERS
# ------------------------------
def build_pipeline(model_entry):
    """Create a text-classification pipeline for a model entry."""
    if isinstance(model_entry, str):
        return pipeline(
            "text-classification",
            model=model_entry,
            tokenizer=model_entry,
            padding=True,
            truncation=True,
        )
    elif isinstance(model_entry, dict):
        model_name = model_entry["name"]
        subfolder = model_entry.get("subfolder")

        if subfolder:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, subfolder=subfolder
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, subfolder=subfolder, trust_remote_code=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            padding=True,
            truncation=True,
        )
    else:
        raise ValueError(f"Unsupported model entry: {model_entry}")


def run_model_inference(model_entry, texts, batch_size=8):
    """Run classifier pipeline and return human-readable predictions."""
    clf = build_pipeline(model_entry)
    preds = []

    # model's label -> numeric 0/1
    numeric_map = {}
    if isinstance(model_entry, dict):
        numeric_map = model_entry.get("label_map", {})

    # numeric 0/1 -> human-readable
    if isinstance(model_entry, dict):
        if "MultilingualSafety_Research" in model_entry["name"]:
            human_map = {0: "not toxic", 1: "toxic"}
        else:  # Ayush-Singh models
            human_map = {0: "no", 1: "yes"}
    else:
        human_map = {0: "safe", 1: "unsafe"}

    total_batches = (len(texts) + batch_size - 1) // batch_size
    model_name = model_entry if isinstance(model_entry, str) else model_entry["name"]

    with tqdm(total=total_batches, desc=f"Processing batches ({model_name.split('/')[-1]})", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            outputs = clf(batch)
            for out in outputs:
                label = out["label"]
                num_pred = numeric_map.get(label, 1 if "1" in label or "unsafe" in label.lower() else 0)
                preds.append(human_map[num_pred])
            pbar.update(1)

    del clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    print("  Evaluating on DAMO-MultiJail...")
    ds = load_dataset("ToxicityPrompts/DAMO-MultiJail", split="test")
    texts = [t for t in ds["text"] if isinstance(t, str)]
    langs = ds["language"]
    y_true = [1 if l > 0 else 0 for l in ds["label"]]

    preds_human = run_model_inference(model_entry, texts)
    # convert back to numeric for metrics
    y_pred = [0 if p in ["not toxic", "no"] else 1 for p in preds_human]

    correct = np.array(y_pred) == np.array(y_true)
    results = {"overall_accuracy": float(correct.mean()), "per_language": {}}
    for lang in set(langs):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        results["per_language"][lang] = float(correct[idxs].mean())

    print(f"    → overall_accuracy: {results['overall_accuracy']:.3f}, sample per_language: {dict(list(results['per_language'].items())[:3])}")
    return results


def eval_csrt(model_entry):
    print("  Evaluating on CSRT...")
    ds = load_dataset("ToxicityPrompts/CSRT", split="code_switch")
    texts = [t for t in ds["text"] if isinstance(t, str)]
    y_true = [1 if l > 0 else 0 for l in ds["label"]]

    preds_human = run_model_inference(model_entry, texts)
    y_pred = [0 if p in ["not toxic", "no"] else 1 for p in preds_human]

    correct = np.array(y_pred) == np.array(y_true)
    accuracy = float(correct.mean())
    print(f"    → accuracy: {accuracy:.3f}")
    return {"accuracy": accuracy}


def eval_rtplx(model_entry):
    print("  Evaluating on RTP-LX...")
    ds = load_dataset("ToxicityPrompts/RTP-LX", split="test")
    texts = [t for t in ds["Prompt"] if isinstance(t, str)]
    langs = ds["Locale"]
    y_true = [1 if ann["Toxicity"] > 1 else 0 for ann in ds["PromptAnnotations"]]

    preds_human = run_model_inference(model_entry, texts)
    y_pred = [0 if p in ["not toxic", "no"] else 1 for p in preds_human]

    overall = compute_metrics(y_true, y_pred, average="macro")
    results = {"overall": overall, "per_locale": {}}

    for lang in set(langs):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        if not idxs:
            continue
        y_sub = [y_true[i] for i in idxs]
        p_sub = [y_pred[i] for i in idxs]
        results["per_locale"][lang] = compute_metrics(y_sub, p_sub, average="macro")

    print(f"    → overall_f1: {overall['f1']:.3f}, sample per_locale: {dict(list(results['per_locale'].items())[:3])}")
    return results


def eval_xsafety(model_entry):
    print("  Evaluating on XSafety...")
    ds = load_dataset("ToxicityPrompts/XSafety", split="test")
    filtered = [(t, c, l) for t, c, l in zip(ds["text"], ds["category"], ds["label"]) if isinstance(t, str)]
    texts, cats, labels = zip(*filtered)

    preds_human = run_model_inference(model_entry, list(texts))
    y_true = [1 if l > 0 else 0 for l in labels]
    y_pred = [0 if p in ["not toxic", "no"] else 1 for p in preds_human]

    correct = np.array(y_pred) == np.array(y_true)
    results = {"overall_accuracy": float(correct.mean()), "per_category": {}}
    for cat in set(cats):
        idxs = [i for i, c in enumerate(cats) if c == cat]
        results["per_category"][cat] = float(correct[idxs].mean())

    print(f"    → overall_accuracy: {results['overall_accuracy']:.3f}, sample per_category: {dict(list(results['per_category'].items())[:3])}")
    return results


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    final_report = {}
    for model_entry in MODELS:
        model_name = model_entry if isinstance(model_entry, str) else model_entry["name"]
        print(f"\n===== Evaluating model: {model_name} =====")
        model_report = {}
        model_report["DAMO-MultiJail"] = eval_damo_multijail(model_entry)
        model_report["CSRT"] = eval_csrt(model_entry)
        model_report["RTP-LX"] = eval_rtplx(model_entry)
        model_report["XSafety"] = eval_xsafety(model_entry)
        final_report[model_name] = model_report

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(REPORT_FILE, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n===== FINAL REPORT =====")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
