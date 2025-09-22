import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay


# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser(description="Train multilingual classifier")

parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-small", help="Model name or path")
parser.add_argument("--dataset_name", type=str, default="multilingual_safety_200k", help="Dataset name (HF hub)")
parser.add_argument("--train_batch_size", type=int, default=12, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=12, help="Eval batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

args = parser.parse_args()


# ---------------------------
# Load dataset
# ---------------------------
print(f"Loading dataset {args.dataset_name}...")
ds = load_dataset(f"Ayush-Singh/{args.dataset_name}")['train']

# ---------------------------
# Map labels
# ---------------------------
ds = ds.map(lambda ex: {"labels": 1 if ex["label"]=="yes" else 0})

# Split train/val (10% validation)
dataset = ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

print("Train size:", len(train_ds))
print("Validation size:", len(val_ds))

# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def preprocess(examples):
    return tokenizer(
        examples["prompt"], 
        truncation=True, 
        padding="max_length", 
        max_length=args.max_length
    )

print("Tokenizing datasets...")
train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------
# Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
model.config.id2label = {0:"no", 1:"yes"}
model.config.label2id = {"no":0, "yes":1}

# ---------------------------
# Metrics with confusion matrix
# ---------------------------
# ---------------------------
# Metrics (no confusion matrix, clean tqdm)
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


training_args = TrainingArguments(
    output_dir=f"./results_{args.dataset_name}",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    save_total_limit=5,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    logging_dir=f"./logs_{args.dataset_name}",
    logging_strategy="steps",
    logging_steps=200,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb",
    run_name=f"{args.dataset_name}_run",
    remove_unused_columns=False,
    fp16=True,
    dataloader_num_workers=4,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    label_names=["labels"],
    disable_tqdm=False,   # keep clean tqdm
    log_level="error",    # stop tqdm spam
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ---------------------------
# Train & Evaluate
# ---------------------------
print("Starting training...")
trainer.train()
results = trainer.evaluate()

print(f"Evaluation results on {args.dataset_name} validation set:")
print(results)

trainer.save_model(f"./final_model_{args.dataset_name}")
