"""
q5 A detailed guide to Pytorch’s nn.Transformer() module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import random

# ====================================
# Positional Encoding
# ====================================
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Encoding from formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


# ====================================
# Transformer Model
# ====================================
class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module."
    """

    def __init__(self, num_tokens, dim_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout_p):
        super().__init__()
        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        transformer_out = self.transformer(
            src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        out = self.out(transformer_out)
        return out

    def get_tgt_mask(self, size) -> torch.Tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def create_pad_mask(self, matrix: torch.Tensor, pad_token: int) -> torch.Tensor:
        return (matrix == pad_token)


# ====================================
# Data Generation
# ====================================
def generate_random_data(n):
    SOS_token = np.array([2])
    EOS_token = np.array([3])
    length = 8
    data = []

    # 1,1,1,1...
    for _ in range(n // 3):
        X = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    # 0,0,0,0...
    for _ in range(n // 3):
        X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])

    # alternating
    for _ in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)
        X[start::2] = 1

        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        X = np.concatenate((SOS_token, X, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))
        data.append([X, y])

    np.random.shuffle(data)
    return data


def batchify_data(data, batch_size=16):
    batches = []
    for idx in range(0, len(data), batch_size):
        if idx + batch_size < len(data):
            batches.append(np.array(data[idx: idx + batch_size]).astype(np.int64))
    return batches


# ====================================
# Training and Validation
# ====================================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

model = Transformer(
    num_tokens=4, dim_model=8, num_heads=2,
    num_encoder_layers=3, num_decoder_layers=3,
    dropout_p=0.1
).to(device)

opt = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pred = model(X, y_input, tgt_mask)
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        val_loss = validation_loop(model, loss_fn, val_dataloader)
        train_loss_list.append(train_loss)
        validation_loss_list.append(val_loss)
        print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    return train_loss_list, validation_loss_list


# ====================================
# Prediction
# ====================================
def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    for _ in range(max_length):
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item()
        next_item = torch.tensor([[next_item]], device=device)
        y_input = torch.cat((y_input, next_item), dim=1)

        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


# ====================================
# Run Training + Prediction
# ====================================
if __name__ == "__main__":
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn,
                                                train_dataloader, val_dataloader, 10)

    import matplotlib.pyplot as plt

    # Plot train vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train loss", color="green")
    plt.plot(validation_loss_list, label="Validation loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    examples = [
        torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
    ]

    for idx, example in enumerate(examples):
        result = predict(model, example)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()
