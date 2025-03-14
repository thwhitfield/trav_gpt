import logging
import os
from pathlib import Path
import tempfile

from hydra import compose, initialize
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from trav_gpt import ROOT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

EVAL_ITERS = 200
LEARNING_RATE = 1e-2
MAX_ITERS = 3000
EMBED_SIZE = 10
EVAL_INTERVAL = 300
CONTEXT_SIZE = 8
BATCH_SIZE = 4
OUTPUT_LENGTH = 1000  # Length of generated text
TEST = False

device = "cpu"


class CharTokenizer:

    def __init__(self):
        pass

    def fit(self, text):

        # Get the unique set of characters
        chars = list(sorted(set(text)))

        # assign each character a number
        self.char_to_int_mapping = {c: i for i, c in enumerate(chars)}
        self.int_to_char_mapping = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, string):
        mapping = self.char_to_int_mapping
        return [mapping[c] for c in string]

    def decode(self, token_ids):
        mapping = self.int_to_char_mapping
        return "".join(mapping[tok] for tok in token_ids)


def get_batch(split):
    """This will convert the input tensor (of the whole text) into the appropriate
    inputs and target labels (x and y)
    """

    data = train if split == "train" else test

    # Grab the starting points. This returns a tensor of shape (BATCH_SIZE,)
    ix = torch.randint(0, len(data) - CONTEXT_SIZE - 1, (BATCH_SIZE,))

    # Once I've grabbed those starting points, then I need to just grab the contexts associated with
    # each one (and also the targets, which will be shifted over by 1)
    x = torch.stack([data[ix[i] : ix[i] + CONTEXT_SIZE] for i in range(BATCH_SIZE)])
    y = torch.stack([data[ix[i] + 1 : ix[i] + CONTEXT_SIZE + 1] for i in range(BATCH_SIZE)])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size=10):
        super().__init__()

        # The embedding dim needs to be the same size as the vocab, because that's the
        # output of this step. It should output the logit associated with each possible
        # character.

        # If I wanted to use a different embedding dimension, then I'd need to first
        # embed the characters to that dimension, then have an additional step which
        # generates the output logits associated with each character.
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )

        self.fc1 = nn.Linear(embed_size, vocab_size)

    def forward(self, x, targets=None):

        logits = self.token_embedding_table(x)  # (B,T,E)
        logits = self.fc1(logits)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # To calculate the loss across the whole batch, we just reshape the
            # logits such that the batches are basically combined. Then we calculate the
            # loss on each of the individual token predictions.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=50):

        for _ in range(max_new_tokens):
            logits, loss = self(
                idx
            )  # (B,T,C) where B = batch size, T = context size, C = vocabulary size
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # Perform softmax on the C dimension
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=EMBED_SIZE
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=CONTEXT_SIZE, embedding_dim=EMBED_SIZE
        )

        self.fc1 = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if TEST:
            logging.info(f"idx shape: {idx.shape}")
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)

        # I guess if we add the (T,C) shape pos_emb to the (B,T,C) shape tok_emb, then torch must
        # know to properly add the pos emb to just the T,C part of each tensor
        x = tok_emb + pos_emb

        if TEST:
            logging.info(f"x shape: {x.shape}")

        x = self.fc1(x)

        logits = x  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # To calculate the loss across the whole batch, we just reshape the
            # logits such that the batches are basically combined. Then we calculate the
            # loss on each of the individual token predictions.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=50):

        for _ in range(max_new_tokens):
            # crop idx to the last CONTEXT_SIZE tokens
            idx_cond = idx[:, -CONTEXT_SIZE:]

            logits, loss = self(
                idx_cond
            )  # (B,T,C) where B = batch size, T = context size, C = vocabulary size
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # Perform softmax on the C dimension
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


if __name__ == "__main__":

    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
    cfg.paths.root = ROOT_DIR

    # Set up MLflow
    mlflow.set_tracking_uri("file:" + os.path.join(ROOT_DIR, "mlruns"))
    mlflow.set_experiment("gpt2_training")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("embed_size", EMBED_SIZE)
        mlflow.log_param("context_size", CONTEXT_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_iters", MAX_ITERS)

        # Log config as artifact with a descriptive name
        config_path = os.path.join(tempfile.gettempdir(), "config.yaml")
        OmegaConf.save(cfg, config_path)
        mlflow.log_artifact(config_path, "config")

        # Read in the text data
        text_path = Path(cfg.paths.external) / "input.txt"

        logging.info(f"Read text data from {text_path}")
        with open(text_path, "r") as f:
            text = f.read()

        # Fit the character level tokenizer
        logging.info(f"Fit the character level tokenizer")
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        # Log vocabulary size
        mlflow.log_param("vocab_size", tokenizer.vocab_size)

        # Encode the input text data as tokens
        logging.info(f"Encode the input text data using the tokenizer")
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Get the train/test split
        logging.info(f"Get the train test split")
        TRAIN_RATIO = 0.9
        mlflow.log_param("train_ratio", TRAIN_RATIO)
        n = int(TRAIN_RATIO * len(data))

        # Split the data first into the train and test datasets
        # There's certainly a better way of doing this with textual data, but we'll do it like this for now.
        train = data[:n]
        test = data[n:]

        logging.info(f"lengths - data: {len(data)}, train: {len(train)}, test: {len(test)}")

        logging.info(
            f"Initialize GPT2 model with vocab_size: {tokenizer.vocab_size} and embed_size: {EMBED_SIZE}"
        )
        model = GPTLanguageModel(vocab_size=tokenizer.vocab_size)
        model = model.to(device)

        logging.info(f"Set up optimizer with learning rate: {LEARNING_RATE}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        logging.info(f"Start model training")
        if TEST:
            MAX_ITERS = 1
        for iter in range(MAX_ITERS):

            if iter % EVAL_ITERS == 0:
                losses = estimate_loss(model)
                logging.info(
                    f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['val']:.4f}"
                )

                # Log metrics to MLflow
                mlflow.log_metrics(
                    {"train_loss": float(losses["train"]), "val_loss": float(losses["val"])},
                    step=iter,
                )

            xb, yb = get_batch("train")

            logits, loss = model(xb, yb)
            optimizer.zero_grad()  # zero out the previous gradients
            loss.backward()  # Backpropagate the loss through the NN
            optimizer.step()  # Update the model parameters using those gradients

        # Generate longer text output
        logging.info(f"Generating {OUTPUT_LENGTH} characters of sample text")
        output = model.generate(
            torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=OUTPUT_LENGTH
        )[0].tolist()
        sample_text = tokenizer.decode(output)

        # Save and log generated text as artifact with a descriptive name
        output_path = os.path.join(tempfile.gettempdir(), "sample_output.txt")
        with open(output_path, "w") as f:
            f.write(sample_text)
        mlflow.log_artifact(output_path, "outputs")

        # Log final loss values
        final_losses = estimate_loss(model)
        mlflow.log_metrics(
            {
                "final_train_loss": float(final_losses["train"]),
                "final_val_loss": float(final_losses["val"]),
            }
        )

        logging.info(f"Sample text preview: {sample_text[:100]}...")
        logging.info(
            f"Training complete. Full generated text saved as MLflow artifact 'sample_output.txt'"
        )
