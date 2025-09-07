import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--use_lora_8bit", action="store_true", help="Enable LoRA fine-tuning")
args = parser.parse_args()

# ---------------------------
# Dataset
# ---------------------------
ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)['train']

def simplify(example):
    if example["data_type"] in ["vanilla_harmful", "adversarial_harmful"]:
        example["data_type"] = "harmful"
    else:
        example["data_type"] = "benign"
    return example

ds = ds.map(simplify)
ds = ds.map(lambda ex: {"text": ex["vanilla"], "label": 1 if "harmful" in ex["data_type"] else 0})

# 👇 take only random 20k samples for training
ds = ds.shuffle(seed=42).select(range(40000))

dataset = ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

eval_ds = load_dataset("Ayush-Singh/qualifire", split="test")
label_map = {"benign": 0, "jailbreak": 1}
eval_ds = eval_ds.map(lambda ex: {"label": label_map[ex["label"]]})

# ---------------------------
# Tokenizer
# ---------------------------
model_name = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

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
# Model + LoRA + 8-bit
# ---------------------------
lora_config = None
if args.use_lora_8bit:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        load_in_8bit=True,
        num_labels=2,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["Wi", "Wo", "dense","Wqkv","classifier"],
    )

else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# attach label metadata
id2label = {0: "benign", 1: "harmful"}
label2id = {"benign": 0, "harmful": 1}
model.config.id2label = id2label
model.config.label2id = label2id

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

keep_cols = ["input_ids", "attention_mask", "labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

# ---------------------------
# Training
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=2500,        # 👈 evaluate every 2.5k steps
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    remove_unused_columns=False,
    fp16=True,
    dataloader_num_workers=4,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    peft_config=lora_config
)

trainer.train()
results = trainer.evaluate(eval_dataset=eval_ds)
print("Evaluation on Ayush-Singh/qualifire:", results)
