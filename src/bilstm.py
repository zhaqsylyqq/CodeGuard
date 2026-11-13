"""A small sequence baseline, trained only on the project's training split."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import tokenize

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from experiment_utils import DEFAULT_DATA_DIR, load_examples, run_metadata, save_report, seed_everything


def code_tokens(code):
    try:
        tokens = []
        for item in tokenize.generate_tokens(io.StringIO(code).readline):
            if item.type == tokenize.INDENT:
                tokens.append("<INDENT>")
            elif item.type == tokenize.DEDENT:
                tokens.append("<DEDENT>")
            elif item.type in {tokenize.NEWLINE, tokenize.NL}:
                tokens.append("<NL>")
            elif item.type not in {tokenize.ENDMARKER, tokenize.ENCODING}:
                tokens.append(item.string)
        return tokens
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.findall(r"\n|[ \t]+|\w+|[^\w\s]", code)


def build_vocab(texts, max_vocab=20000):
    if max_vocab < 2:
        raise ValueError("max_vocab must be at least 2")
    counts = Counter(token for text in texts for token in code_tokens(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counts.most_common(max_vocab - 2):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class CodeDataset(Dataset):
    def __init__(self, examples, vocab, max_length):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.items = [
            (torch.tensor([vocab.get(t, 1) for t in code_tokens(text)[:max_length]] or [1]), float(label))
            for text, label in zip(examples.text, examples.label)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_batch(items):
    sequences, labels = zip(*items)
    return pad_sequence(sequences, batch_first=True), torch.tensor([len(x) for x in sequences]), torch.tensor(labels)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1))

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=1)).squeeze(1)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probabilities = []
    for tokens, lengths, _ in loader:
        probabilities.extend(model(tokens.to(device), lengths).sigmoid().cpu().tolist())
    return probabilities


def run(args):
    seed_everything(args.seed)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    train = load_examples(args.data_dir, "train", args.limit_pairs, args.seed)
    val = load_examples(args.data_dir, "val", args.limit_pairs, args.seed)
    vocab = build_vocab(train.text, args.max_vocab)
    train_loader = DataLoader(CodeDataset(train, vocab, args.max_length), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(CodeDataset(val, vocab, args.max_length), batch_size=args.batch_size,
                            collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"vocab_size": len(vocab), "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim}
    model = BiLSTM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_loss, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for tokens, lengths, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(tokens.to(device), lengths), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        probabilities = predict(model, val_loader, device)
        val_loss = nn.functional.binary_cross_entropy(torch.tensor(probabilities), torch.tensor(val.label.to_numpy(), dtype=torch.float32)).item()
        row = {"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss}
        history.append(row)
        print(row)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab": vocab,
                        "max_length": args.max_length, "epoch": epoch}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = {"task": "detection", "model": "bilstm", "split": "val", "seed": args.seed,
                "best_epoch": checkpoint["epoch"], "settings": vars(args), "history": history,
                **run_metadata(args)}
    report = save_report(output, val, predict(model, val_loader, device), metadata)
    print(json.dumps(report, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="data/processed/small or data/processed/full")
    p.add_argument("--output", default="runs/bilstm")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-pairs", type=int)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
