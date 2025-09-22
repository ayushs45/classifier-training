import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch


# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser(description="Train multilingual classifier with 4-bit quantization")

parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-small", help="Model name or path")
parser.add_argument("--dataset_name", type=str, default="multilingual_safety_200k", help="Dataset name (HF hub)")
parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size (reduced for 4-bit)")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (higher for LoRA)")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

args = parser.parse_args()


# ---------------------------
# 4-bit quantization config
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization for better performance
    bnb_4bit_quant_type="nf4",       # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for better stability
)

print("Using 4-bit quantization with the following config:")
print(f"- Quantization type: {bnb_config.bnb_4bit_quant_type}")
print(f"- Double quantization: {bnb_config.bnb_4bit_use_double_quant}")
print(f"- Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")


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
# Tokenizer with padding token fix
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Fix for models without padding token (like Qwen, Llama, etc.)
if tokenizer.pad_token is None:
    # Try to use eos_token as pad_token
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    else:
        # Fallback: add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added new pad_token: [PAD]")

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
# Model with 4-bit quantization
# ---------------------------
print("Loading model with 4-bit quantization...")
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, 
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.bfloat16,
)

model.config.id2label = {0:"no", 1:"yes"}
model.config.label2id = {"no":0, "yes":1}

# Resize token embeddings if we added a new pad token
if tokenizer.pad_token == '[PAD]':
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized token embeddings to {len(tokenizer)} tokens")

# Set pad_token_id in model config to match tokenizer
model.config.pad_token_id = tokenizer.pad_token_id
print(f"Set model pad_token_id to: {model.config.pad_token_id}")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ---------------------------
# LoRA Configuration
# ---------------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["query", "value", "key", "dense"],  # Common attention modules
    bias="none",
)

print("Applying LoRA with the following config:")
print(f"- Rank (r): {args.lora_r}")
print(f"- Alpha: {args.lora_alpha}")
print(f"- Dropout: {args.lora_dropout}")
print(f"- Target modules: {lora_config.target_modules}")

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------
# Training Arguments (adjusted for quantized training)
# ---------------------------
training_args = TrainingArguments(
    output_dir=f"./results_{args.dataset_name}_4bit",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    save_total_limit=3,  # Reduced to save disk space
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    logging_dir=f"./logs_{args.dataset_name}_4bit",
    logging_strategy="steps",
    logging_steps=200,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb",
    run_name=f"{args.dataset_name}_4bit_run",
    remove_unused_columns=False,
    bf16=True,  # Use bfloat16 instead of fp16 for better stability with quantization
    dataloader_num_workers=4,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    label_names=["labels"],
    disable_tqdm=False,
    log_level="error",
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    optim="paged_adamw_8bit",  # Use 8-bit optimizer for additional memory savings
    max_grad_norm=1.0,  # Gradient clipping for stability
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
print("Starting 4-bit quantized training...")
print(f"Tokenizer pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"Model pad_token_id: {model.config.pad_token_id}")
print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

trainer.train()
results = trainer.evaluate()

print(f"Evaluation results on {args.dataset_name} validation set:")
print(results)

# ---------------------------
# Save the model (LoRA adapters only)
# ---------------------------
output_dir = f"./final_model_{args.dataset_name}_4bit_lora"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"LoRA adapters and tokenizer saved to {output_dir}")
print("Note: Only LoRA adapters are saved. To use the model, load the base model with quantization and apply these adapters.")

# ---------------------------
# Memory usage summary
# ---------------------------
if torch.cuda.is_available():
    print(f"GPU Memory after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
