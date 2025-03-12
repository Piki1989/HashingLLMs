import time
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


# --- Custom Hash Trick Embedding Layer ---
class HashTrickEmbedding(nn.Module):
    """
    A simple hash trick embedding layer that maps token IDs into hash buckets.
    This static implementation uses a modulo operation for hashing.

    Args:
        vocab_size (int): The original vocabulary size.
        num_buckets (int): Number of hash buckets.
        embedding_dim (int): Dimensionality of the output embeddings.
    """

    def __init__(self, vocab_size, num_buckets, embedding_dim):
        super(HashTrickEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        # Create an embedding layer for the hash buckets.
        self.bucket_embeddings = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, token_ids):
        # Use a simple modulo hash: map token IDs into buckets.
        hashed_ids = token_ids % self.num_buckets
        embeddings = self.bucket_embeddings(hashed_ids)
        return embeddings


# --- Function to Modify GPT-2 Embedding Layer ---
def modify_model_embeddings(model, tokenizer, num_buckets):
    """
    Replace GPT-2's word token embedding (wte) with our HashTrickEmbedding.
    """
    embedding_dim = model.config.n_embd  # GPT-2 hidden size (e.g., 768 for GPT-2 small)
    vocab_size = tokenizer.vocab_size
    # Create our hash trick embedding layer.
    new_embedding = HashTrickEmbedding(vocab_size, num_buckets, embedding_dim)
    # Replace GPT-2's embedding layer with our custom layer.
    model.transformer.wte = new_embedding
    return model


# --- Data Preprocessing Functions ---
def tokenize_and_group(examples, tokenizer, block_size):
    """
    Tokenize texts and group tokens into blocks of fixed size.
    """
    tokenized = tokenizer(examples["text"])
    # Concatenate all token ids.
    concatenated = sum(tokenized["input_ids"], [])
    total_length = len(concatenated)
    # Ensure total_length is a multiple of block_size.
    total_length = (total_length // block_size) * block_size
    input_ids = [
        concatenated[i: i + block_size] for i in range(0, total_length, block_size)
    ]
    return {"input_ids": input_ids}


# --- Experiment Function ---
def run_experiment(use_hash_embedding: bool, num_buckets: int = 1024):
    """
    Runs a training and evaluation experiment on WikiText-2 using GPT-2.
    If use_hash_embedding is True, replaces the embedding layer with a hash trick layer.

    Returns a dictionary with metrics: eval_loss, perplexity, and training_time.
    """
    model_name = "gpt2"  # Pre-trained GPT-2 model.
    block_size = 128  # Maximum sequence length for training.
    num_train_epochs = 1  # For demonstration; adjust as needed.
    per_device_batch_size = 2  # Adjust based on your hardware.

    # Load tokenizer and set the padding token to eos_token.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Fix for padding token issue.

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Optionally modify embeddings.
    variant = "HashTrick" if use_hash_embedding else "Original"
    if use_hash_embedding:
        model = modify_model_embeddings(model, tokenizer, num_buckets)

    # Load and preprocess WikiText-2 dataset.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_group(examples, tokenizer, block_size),
        batched=True,
        remove_columns=["text"],
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Data collator (for causal language modeling, no masking).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=f"./{variant}_gpt2_experiment",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        logging_dir=f"./logs_{variant}",
        learning_rate=5e-5,
        report_to="none"  # Disable logging to W&B or others.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and time the training process.
    print(f"\nStarting training for {variant} model...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training time for {variant} model: {training_time:.2f} seconds")

    # Evaluate the model.
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = np.exp(eval_loss)

    metrics = {
        "variant": variant,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "training_time": training_time,
    }

    # --- Generate Sample Text for Qualitative Comparison ---
    input_text = "Once upon a time"
    # Ensure input_ids and attention_mask are on the same device as the model.
    device = model.device  # Likely "mps" on Apple Silicon if available.
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # Create an attention mask (all ones in this simple case).
    attention_mask = torch.ones_like(input_ids).to(device)

    sample_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    metrics["sample_text"] = generated_text

    print(f"\n{variant} Model Metrics:")
    print(f"Evaluation Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Sample Generation: {generated_text}\n")

    return metrics


# --- Main Function to Run Both Experiments ---
def main():
    print("Running experiments to compare Original vs HashTrick Embedding in GPT-2...\n")
    results_original = run_experiment(use_hash_embedding=False)
    results_hashtrick = run_experiment(use_hash_embedding=True, num_buckets=1024)

    print("Final Comparison of Metrics:")
    for res in [results_original, results_hashtrick]:
        print(f"Variant: {res['variant']}")
        print(f"  Eval Loss: {res['eval_loss']:.4f}")
        print(f"  Perplexity: {res['perplexity']:.2f}")
        print(f"  Training Time: {res['training_time']:.2f} seconds")
        print(f"  Sample Generation: {res['sample_text']}\n")


if __name__ == "__main__":
    main()
