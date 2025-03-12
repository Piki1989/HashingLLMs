import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# Load model and tokenizer from the checkpoint
checkpoint_dir = "/Users/michalpikus/Projects/HashingLLMs/results_microsoft/Phi-4-mini-instruct_one_hot/checkpoint-1000"  # Replace with your actual path
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

# (Optional) Apply custom embeddings if needed
# model = apply_one_hot_trick(model)  # or apply_hash_trick(model)

# Load and preprocess the evaluation dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_data(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(preprocess_data, batched=True)
eval_dataset = tokenized_datasets["validation"].select(range(200))

# Set up the Trainer for evaluation only
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=1,
    do_train=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
)

# Run evaluation
eval_results = trainer.evaluate()
perplexity = np.exp(eval_results["eval_loss"])

print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {perplexity:.2f}")
