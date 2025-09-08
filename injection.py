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
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-large", help="Model to use")
parser.add_argument("--train_batch_size", type=int, default=12, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=12, help="Evaluation batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy (steps/epoch)")
parser.add_argument("--eval_steps", type=int, default=2500, help="Steps between evaluations if strategy=steps")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
args = parser.parse_args()

# ---------------------------
# Dataset
# ---------------------------
ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)['train']

def simplify(example):
    example["data_type"] = "harmful" if example["data_type"] in ["vanilla_harmful", "adversarial_harmful"] else "benign"
    return example

ds = ds.map(simplify)
ds = ds.map(lambda ex: {"text": ex["vanilla"], "label": 1 if ex["data_type"] == "harmful" else 0})

# Take random 40k samples
ds = ds.shuffle(seed=42).select(range(40000))
dataset = ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

eval_ds = load_dataset("Ayush-Singh/qualifire", split="test")
label_map = {"benign": 0, "jailbreak": 1}
eval_ds = eval_ds.map(lambda ex: {"label": label_map[ex["label"]]})

# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
eval_ds = eval_ds.rename_column("label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------
# Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

# Attach label metadata
id2label = {0: "benign", 1: "harmful"}
label2id = {"benign": 0, "harmful": 1}
model.config.id2label = id2label
model.config.label2id = label2id

# ---------------------------
# Metrics with confusion matrix
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "harmful"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# Remove unused columns
keep_cols = ["input_ids", "attention_mask", "labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

# ---------------------------
# Training
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    save_strategy="no",   # <-- no checkpoints
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    logging_dir="./logs",
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    remove_unused_columns=False,
    fp16=True,
    dataloader_num_workers=4,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate(eval_dataset=eval_ds)
print("Evaluation on Ayush-Singh/qualifire:", results)

# Save final model only at the end
trainer.save_model("./final_model")
