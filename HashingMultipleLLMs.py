import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import time
import numpy as np
import torch.nn as nn
import hashlib


class HashEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_buckets=10000):
        """
        Implements a hashed embedding lookup table.
        :param vocab_size: The original vocabulary size.
        :param embed_dim: The embedding dimension.
        :param num_buckets: The number of hash buckets (smaller than vocab_size).
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)

    def hash_token(self, token_id):
        """Apply a simple hash function to map token_id to a bucket."""
        return int(hashlib.md5(str(token_id).encode()).hexdigest(), 16) % self.num_buckets

    def forward(self, input_ids):
        """Map input tokens to hashed embeddings."""
        hashed_ids = torch.tensor([self.hash_token(t.item()) for t in input_ids.view(-1)]).to(input_ids.device)
        return self.embedding(hashed_ids).view(*input_ids.shape, -1)

def apply_hash_trick(model):
    print("Applying Hash Trick Embeddings...")

    num_buckets = 5000  # Example hash bucket size (tune this as needed)

    if hasattr(model, "transformer"):  # GPT-2 and similar architectures
        model.transformer.wte = HashEmbedding(model.config.vocab_size, model.config.hidden_size, num_buckets)
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):  # Facebook OPT models
        model.model.decoder.embed_tokens = HashEmbedding(model.config.vocab_size, model.config.hidden_size, num_buckets)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = HashEmbedding(model.config.vocab_size, model.config.hidden_size, num_buckets)
    else:
        raise ValueError("Unsupported model architecture for Hash Trick")

    return model


class OneHotEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Implements a one-hot encoding based embedding layer.
        """
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim, bias=False)

    def forward(self, input_ids):
        """Convert input_ids to one-hot representations and project to embedding space."""
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.embedding.in_features).float()
        return self.embedding(one_hot)


def apply_one_hot_trick(model):
    """
    Replace the model's word embedding layer with a one-hot encoding based embedding layer.
    """
    print("Applying One-Hot Trick Embeddings...")
    if hasattr(model, "transformer"):  # GPT-2 and similar architectures
        model.transformer.wte = OneHotEmbedding(model.config.vocab_size, model.config.hidden_size)
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):  # Facebook OPT models
        model.model.decoder.embed_tokens = OneHotEmbedding(model.config.vocab_size, model.config.hidden_size)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = OneHotEmbedding(model.config.vocab_size, model.config.hidden_size)
    else:
        raise ValueError("Unsupported model architecture for One Hot Trick")
    return model


def load_model(model_name, embedding_type="original"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if embedding_type == "hash":
        model = apply_hash_trick(model)
    elif embedding_type == "one_hot":
        model = apply_one_hot_trick(model)

    return model, tokenizer


def train_and_evaluate(model, tokenizer, model_name, embedding_type):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD if none exists

    def preprocess_data(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Ensure all sequences have the same length
            max_length=512,  # Adjust based on model (OPT has no default max length)
        )
        tokenized["labels"] = tokenized["input_ids"].copy()  # Set labels for loss calculation
        return tokenized

    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}_{embedding_type}",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].select(range(1000)),
        eval_dataset=tokenized_datasets["validation"].select(range(200)),
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    eval_results = trainer.evaluate()
    perplexity = np.exp(eval_results["eval_loss"])

    print(f"\nModel: {model_name} ({embedding_type})")
    print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Training Time: {training_time:.2f} sec\n")

    return eval_results, perplexity, training_time


if __name__ == "__main__":
    model_names = ["gpt2","facebook/opt-1.3b","microsoft/Phi-4-mini-instruct"]
    embedding_types = ["orginal","hash","one_hot"]

    results = {}
    for model_name in model_names:
        for embedding_type in embedding_types:
            print(f"\nRunning {model_name} with {embedding_type} embeddings...")
            model, tokenizer = load_model(model_name, embedding_type=embedding_type)
            results[(model_name, embedding_type)] = train_and_evaluate(model, tokenizer, model_name, embedding_type)

    print("\nFinal Results Comparison:")
    for (model_name, embedding_type), (eval_results, perplexity, training_time) in results.items():
        print(f"{model_name} ({embedding_type}): Perplexity={perplexity:.2f}, Time={training_time:.2f}s")
