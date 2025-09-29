import json
import gc
import numpy as np
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# CONFIG
# ------------------------------
MODELS = [
    # {
    #     "name": "repelloai/MultilingualSafety_Research",
    #     "subfolder": "xlm-roberta-large-v1",
    #     "label_map": {0: "not toxic", 1: "toxic"},
    # },
    # {
    #     "name": "Ayush-Singh/mmBert_final_model_multilingual_safety_200k",
    #     "label_map": {0: "no", 1: "yes"},
    # },
    {
        "name": "Ayush-Singh/qwen-prompt-guard",
        "label_map": {0: 0, 1: 1},
    },
]

REPORT_FILE = "safety_eval_report.json"

# Standardized toxic labels for consistency
TOXIC_LABELS = {"toxic", "yes", "TOXIC", "YES", "1", 1}
SAFE_LABELS = {"not toxic", "no", "NOT_TOXIC", "NO", "0", 0}


# ------------------------------
# HELPERS
# ------------------------------
def build_pipeline(model_entry):
    """Create a pipeline given either a string or dict with subfolder."""
    try:
        if isinstance(model_entry, str):
            return pipeline(
                "text-classification",
                model=model_entry,
                tokenizer=model_entry,
                padding=True,
                truncation=True,
                device=0 if torch.cuda.is_available() else -1
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
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            raise ValueError(f"Unsupported model entry: {model_entry}")
    except Exception as e:
        logger.error(f"Error building pipeline for {model_entry}: {e}")
        raise


def normalize_prediction(pred, label_map=None):
    """Normalize prediction to binary (0=safe, 1=toxic)"""
    if label_map and pred in label_map:
        pred = label_map[pred]
    
    # Handle LABEL_X format
    if isinstance(pred, str) and pred.startswith("LABEL_"):
        try:
            label_int = int(pred.split("_")[1])
            if label_map:
                pred = label_map.get(label_int, label_int)
            else:
                pred = label_int
        except (IndexError, ValueError):
            pass
    
    # Convert to binary
    if pred in TOXIC_LABELS:
        return 1
    elif pred in SAFE_LABELS:
        return 0
    else:
        # Fallback: assume numeric or attempt conversion
        try:
            return int(float(pred) > 0.5)
        except:
            logger.warning(f"Unknown prediction format: {pred}, assuming safe (0)")
            return 0


def run_model_inference(model_entry, texts, batch_size=16):
    """Run classifier pipeline using KeyDataset for efficient batching."""
    try:
        clf = build_pipeline(model_entry)
        
        label_map = model_entry.get("label_map", {}) if isinstance(model_entry, dict) else {}
        model_name = model_entry if isinstance(model_entry, str) else model_entry["name"]

        # Create a dataset from texts for KeyDataset usage
        # Handle None values by converting to empty strings
        clean_texts = [str(text) if text is not None else "" for text in texts]
        text_dataset = Dataset.from_dict({"text": clean_texts})
        
        preds = []
        
        # Use KeyDataset for efficient batching
        total_samples = len(clean_texts)
        with tqdm(total=total_samples, desc=f"Processing samples ({model_name.split('/')[-1]})", unit="sample") as pbar:
            try:
                for output in clf(KeyDataset(text_dataset, "text"), batch_size=batch_size):
                    label = output["label"]
                    normalized_pred = normalize_prediction(label, label_map)
                    preds.append(normalized_pred)
                    pbar.update(1)
            except Exception as e:
                logger.warning(f"Error during pipeline processing: {e}")
                # Fill remaining predictions with safe (0) if processing failed
                remaining = total_samples - len(preds)
                preds.extend([0] * remaining)

        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return preds
    
    except Exception as e:
        logger.error(f"Error in model inference: {e}")
        # Return safe predictions as fallback
        return [0] * len(texts)


def compute_metrics(y_true, y_pred, average="macro"):
    """Compute classification metrics with error handling."""
    try:
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


# ------------------------------
# DATASET EVALUATIONS
# ------------------------------
def eval_damo_multijail(model_entry):
    """Evaluate on DAMO-MultiJail dataset - all unsafe prompts."""
    print("  Evaluating on DAMO-MultiJail...")
    try:
        ds = load_dataset("ToxicityPrompts/DAMO-MultiJail", split="test")
        texts = [str(t) for t in ds["text"] if t is not None]
        langs = ds["language"]

        if not texts:
            logger.warning("No valid texts found in DAMO-MultiJail")
            return {"overall_accuracy": 0.0, "per_language": {}}

        preds = run_model_inference(model_entry, texts)
        y_true = [1] * len(texts)  # all unsafe
        y_pred = preds

        correct = np.array(y_pred) == np.array(y_true)
        results = {"overall_accuracy": float(correct.mean()), "per_language": {}}
        
        for lang in set(langs):
            idxs = [i for i, l in enumerate(langs) if l == lang and i < len(y_pred)]
            if idxs:
                results["per_language"][lang] = float(correct[idxs].mean())

        print(f"    → overall_accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating DAMO-MultiJail: {e}")
        return {"overall_accuracy": 0.0, "per_language": {}, "error": str(e)}


def eval_csrt(model_entry):
    """Evaluate on CSRT dataset - code_switch split, all unsafe."""
    print("  Evaluating on CSRT...")
    try:
        ds = load_dataset("ToxicityPrompts/CSRT", split="code_switch")
        texts = [str(t) for t in ds["text"] if t is not None]

        if not texts:
            logger.warning("No valid texts found in CSRT")
            return {"accuracy": 0.0}

        preds = run_model_inference(model_entry, texts)
        y_true = [1] * len(texts)  # all unsafe
        y_pred = preds

        acc = float((np.array(y_pred) == np.array(y_true)).mean())
        print(f"    → accuracy: {acc:.3f}")
        return {"accuracy": acc}
    
    except Exception as e:
        logger.error(f"Error evaluating CSRT: {e}")
        return {"accuracy": 0.0, "error": str(e)}


def eval_rtplx(model_entry):
    """Evaluate on RTP-LX dataset - mixed safe/unsafe based on Toxicity score."""
    print("  Evaluating on RTP-LX...")
    try:
        ds = load_dataset("ToxicityPrompts/RTP-LX", split="test")
        
        # Filter and extract data with error handling
        valid_data = []
        for prompt, locale, annotations in zip(ds["Prompt"], ds["Locale"], ds["PromptAnnotations"]):
            if prompt is not None and locale is not None and annotations is not None:
                try:
                    toxicity_score = annotations.get("Toxicity", 0)
                    label = 1 if toxicity_score > 1 else 0
                    valid_data.append((str(prompt), locale, label))
                except (AttributeError, TypeError):
                    continue
        
        if not valid_data:
            logger.warning("No valid data found in RTP-LX")
            return {"overall": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, "per_locale": {}}

        texts, langs, y_true = zip(*valid_data)
        
        preds = run_model_inference(model_entry, list(texts))
        y_pred = preds[:len(y_true)]  # Ensure same length

        overall = compute_metrics(y_true, y_pred, average="macro")
        results = {"overall": overall, "per_locale": {}}

        for lang in set(langs):
            idxs = [i for i, l in enumerate(langs) if l == lang]
            if idxs:
                y_sub = [y_true[i] for i in idxs]
                p_sub = [y_pred[i] for i in idxs]
                results["per_locale"][lang] = compute_metrics(y_sub, p_sub, average="macro")

        print(f"    → overall F1: {overall['f1']:.3f}")
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating RTP-LX: {e}")
        return {"overall": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, "per_locale": {}, "error": str(e)}


def eval_xsafety(model_entry):
    """Evaluate on XSafety dataset - all unsafe prompts."""
    print("  Evaluating on XSafety...")
    try:
        ds = load_dataset("ToxicityPrompts/XSafety", split="test")

        # Filter valid data - using both language and category
        valid_data = [(str(t), l, c) for t, l, c in zip(ds["text"], ds["language"], ds["category"]) 
                     if t is not None and l is not None and c is not None]
        
        if not valid_data:
            logger.warning("No valid data found in XSafety")
            return {"overall_accuracy": 0.0, "per_language": {}, "per_category": {}}

        texts, langs, cats = zip(*valid_data)

        preds = run_model_inference(model_entry, list(texts))
        y_true = [1] * len(texts)  # all unsafe
        y_pred = preds

        correct = np.array(y_pred) == np.array(y_true)
        results = {
            "overall_accuracy": float(correct.mean()), 
            "per_language": {},
            "per_category": {}
        }
        
        # Per-language breakdown
        for lang in set(langs):
            idxs = [i for i, l in enumerate(langs) if l == lang and i < len(y_pred)]
            if idxs:
                results["per_language"][lang] = float(correct[idxs].mean())
        
        # Per-category breakdown  
        for cat in set(cats):
            idxs = [i for i, c in enumerate(cats) if c == cat and i < len(y_pred)]
            if idxs:
                results["per_category"][cat] = float(correct[idxs].mean())

        print(f"    → overall_accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating XSafety: {e}")
        return {"overall_accuracy": 0.0, "per_language": {}, "per_category": {}, "error": str(e)}


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    """Main evaluation loop."""
    final_report = {}
    
    for model_entry in MODELS:
        model_name = model_entry["name"] if isinstance(model_entry, dict) else model_entry
        print(f"\n===== Evaluating model: {model_name} =====")
        
        model_report = {}
        try:
            model_report["DAMO-MultiJail"] = eval_damo_multijail(model_entry)
            model_report["CSRT"] = eval_csrt(model_entry)
            model_report["RTP-LX"] = eval_rtplx(model_entry)
            model_report["XSafety"] = eval_xsafety(model_entry)
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            model_report["error"] = str(e)
        
        final_report[model_name] = model_report

        # Cleanup after each model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save report
    try:
        with open(REPORT_FILE, "w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nReport saved to {REPORT_FILE}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

    print("\n===== FINAL REPORT SUMMARY =====")
    for model_name, results in final_report.items():
        print(f"\n{model_name}:")
        for dataset, metrics in results.items():
            if dataset == "error":
                print(f"  ERROR: {metrics}")
            elif isinstance(metrics, dict):
                if "overall_accuracy" in metrics:
                    print(f"  {dataset}: {metrics['overall_accuracy']:.3f}")
                elif "accuracy" in metrics:
                    print(f"  {dataset}: {metrics['accuracy']:.3f}")
                elif "overall" in metrics and "f1" in metrics["overall"]:
                    print(f"  {dataset}: F1={metrics['overall']['f1']:.3f}")


if __name__ == "__main__":
    main()
