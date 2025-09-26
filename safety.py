import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
parser.add_argument("--hub_repo", type=str, default=None, help="Hugging Face Hub repo (e.g. username/model-name)")
parser.add_argument("--push_steps", type=int, default=0, help="Push to Hub every N steps (0 = disable)")

args = parser.parse_args()


# ---------------------------
# Load dataset
# ---------------------------
print(f"Loading dataset {args.dataset_name}...")
ds = load_dataset(args.dataset_name)['train']

# ---------------------------
# Map labels
# ---------------------------
def map_labels(example):
    label = example["label"]
    if isinstance(label, str):
        if label.lower() in ["yes", "1", "true", "positive"]:
            return {"labels": 1}
        elif label.lower() in ["no", "0", "false", "negative"]:
            return {"labels": 0}
        else:
            raise ValueError(f"Unknown string label: {label}")
    elif isinstance(label, (int, float)):
        return {"labels": int(label)}
    else:
        raise ValueError(f"Unknown label type: {type(label)} with value: {label}")

ds = ds.map(map_labels)

dataset = ds.train_test_split(test_size=0.02, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

print("Train size:", len(train_ds))
print("Validation size:", len(val_ds))


# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    else:
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
# Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

if tokenizer.pad_token == '[PAD]':
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized token embeddings to {len(tokenizer)} tokens")

model.config.id2label = {0: "no", 1: "yes"}
model.config.label2id = {"no": 0, "yes": 1}
model.config.pad_token_id = tokenizer.pad_token_id
print(f"Set model pad_token_id to: {model.config.pad_token_id}")


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
# Improved Push-to-Hub callback
# ---------------------------
class PushToHubCallback(TrainerCallback):
    def __init__(self, repo_name, push_steps):
        self.repo_name = repo_name
        self.push_steps = push_steps

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.push_steps > 0 and state.global_step % self.push_steps == 0 and state.global_step > 0:
            try:
                print(f"🔼 Pushing model to Hub at step {state.global_step} -> {self.repo_name}")
                
                # Create descriptive commit message
                commit_message = f"Training checkpoint at step {state.global_step}"
                
                # Push model and tokenizer with error handling
                model.push_to_hub(
                    self.repo_name, 
                    commit_message=commit_message,
                    safe_serialization=True  # Use safetensors format
                )
                tokenizer.push_to_hub(
                    self.repo_name, 
                    commit_message=commit_message
                )
                print(f"✅ Successfully pushed to Hub at step {state.global_step}")
                
            except Exception as e:
                print(f"❌ Failed to push to Hub at step {state.global_step}: {str(e)}")
                # Continue training even if push fails
                pass

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Push final model at the end of training"""
        if self.repo_name:
            try:
                print(f"🔼 Pushing final model to Hub: {self.repo_name}")
                
                commit_message = f"Final model after {state.global_step} steps"
                
                model.push_to_hub(
                    self.repo_name, 
                    commit_message=commit_message,
                    safe_serialization=True
                )
                tokenizer.push_to_hub(
                    self.repo_name, 
                    commit_message=commit_message
                )
                print(f"✅ Successfully pushed final model to Hub")
                
            except Exception as e:
                print(f"❌ Failed to push final model to Hub: {str(e)}")


# ---------------------------
# Training setup
# ---------------------------
training_args = TrainingArguments(
    output_dir=f"./results_{args.model_name}_{args.dataset_name}",
    eval_strategy="steps",
    eval_steps=1500,
    save_strategy="epoch",
    save_total_limit=5,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    logging_dir=f"./logs_{args.model_name}_{args.dataset_name}",
    logging_strategy="steps",
    logging_steps=200,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb",
    run_name=f"{args.model_name}_{args.dataset_name}_run",
    remove_unused_columns=False,
    fp16=True,
    dataloader_num_workers=4,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    label_names=["labels"],
    disable_tqdm=False,
    log_level="error",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# Attach callback if hub_repo provided
if args.hub_repo:
    trainer.add_callback(PushToHubCallback(repo_name=args.hub_repo, push_steps=args.push_steps))


# ---------------------------
# Train & Evaluate
# ---------------------------
print("Starting training...")
print(f"Tokenizer pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"Model pad_token_id: {model.config.pad_token_id}")

trainer.train()
results = trainer.evaluate()

print(f"Evaluation results on {args.dataset_name} validation set:")
print(results)

# Save final model locally
output_dir = f"./final_model_{args.model_name.replace('/', '_')}_{args.dataset_name}"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Note: Final push to hub is now handled by the callback's on_train_end method
print("Training completed!")
