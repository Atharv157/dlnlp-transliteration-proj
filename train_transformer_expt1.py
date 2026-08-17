# train_transformer_translit.py
"""
Single-file Transformer encoder-decoder + training loop for transliteration.

Usage:
    python train_transformer_translit.py --config path/to/config.yaml

If you want to run programmatically, import main(config_dict).
"""

import argparse
import math
import os
import random
import time
from typing import Tuple, List

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# IMPORTS FROM YOUR PROJECT (adjust module paths if needed)
from src.data.vocabulary import RomanVocabulary, DevanagariVocabulary
from src.data.dataset import TransliterationDataset, collate_fn
from src.data.preprocessor import DataPreprocessor  # uses sampler internally
# If you placed preprocessor in a different module path, adjust imports accordingly.

_mask_cache = {}
# -------------------------
# Utility: positional encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd d_model: last column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


# -------------------------
# Transformer seq2seq model
# -------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # easier shapes: (batch, seq, feature)
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        # src: (batch, src_len)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor = None,
    ):
        # tgt: (batch, tgt_len)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)  # (batch, tgt_len, vocab)
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        return logits


# -------------------------
# Mask helpers
# -------------------------
def make_src_key_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # src: (batch, src_len) -> mask: (batch, src_len) with True where padding
    return (src == pad_idx)


def make_tgt_key_padding_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (tgt == pad_idx)


def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    if sz not in _mask_cache:
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        _mask_cache[sz] = mask
    mask = _mask_cache[sz]
    return mask.to(device) if device else mask

def parallel_scheduled_sampling(model, src, tgt_input, tgt_output, src_key_padding_mask, pad_idx, prob=0.3):
    """Fast alternative to autoregressive - mix teacher forcing with predictions"""
    batch_size, seq_len = tgt_input.shape
    
    # Get predictions in one forward pass (no causal mask for prediction)
    with torch.no_grad():
        # Use a small temperature for more diverse predictions
        pred_logits = model(
            src, 
            tgt_input, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=make_tgt_key_padding_mask(tgt_input, pad_idx),
            tgt_mask=None,  # No causal mask for parallel prediction
            memory_key_padding_mask=src_key_padding_mask
        )
        # Use temperature sampling for more diversity
        temperature = 0.8
        scaled_logits = pred_logits / temperature
        predictions = torch.softmax(scaled_logits, dim=-1).argmax(-1)
    
    # Replace some tokens in tgt_input with predictions
    replace_mask = torch.rand(batch_size, seq_len, device=src.device) < prob
    mixed_input = tgt_input.clone()
    mixed_input[replace_mask] = predictions[replace_mask]
    
    # Final forward pass with proper causal mask for training
    tgt_mask = generate_square_subsequent_mask(seq_len, device=src.device)
    logits = model(
        src,
        mixed_input,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=make_tgt_key_padding_mask(mixed_input, pad_idx),
        tgt_mask=tgt_mask,
        memory_key_padding_mask=src_key_padding_mask
    )
    
    return logits



# -------------------------
# Training / evaluation loops
# -------------------------
def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    dataloader: DataLoader,
    device: torch.device,
    pad_idx: int,
    grad_clip: float,
    teacher_forcing_ratio: float,
    scheduler
):
    model.train()
    total_loss = 0.0
    iters = 0

    for batch in dataloader:
        src, tgt, src_lens, tgt_lens = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # For Transformer: decoder input is tgt_input (without last token)
        # and labels are tgt_output (without first token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]  # expected tokens

        src_key_padding_mask = make_src_key_padding_mask(src, pad_idx=pad_idx)  # (batch, src_len)
        tgt_key_padding_mask = make_tgt_key_padding_mask(tgt_input, pad_idx=pad_idx)

        # teacher forcing: with probability teacher_forcing_ratio we feed the true tgt_input,
        # otherwise we can feed model predictions step-by-step. For simplicity & speed we
        # approximate teacher forcing by mixing: with prob 1 use true tgt_input (standard).
        # If advanced per-step forcing is needed, replace with autoregressive loop.
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)
            logits = model(
                src,
                tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
            )
        else:
            # FAST PATH: Parallel scheduled sampling instead of autoregressive
            logits = parallel_scheduled_sampling(
                model, src, tgt_input, tgt_output, 
                src_key_padding_mask, pad_idx, prob=0.3
            )

        # logits: (batch, tgt_len, vocab)
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_out_flat = tgt_output.contiguous().view(-1)

        loss = criterion(logits_flat, tgt_out_flat)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        iters += 1

    return total_loss / max(1, iters)


def evaluate_one_epoch(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    dataloader: DataLoader,
    device: torch.device,
    pad_idx: int,
    max_decoding_len: int = 200,
    src_vocab: RomanVocabulary = None,
    tgt_vocab: DevanagariVocabulary = None,
):
    model.eval()
    total_loss = 0.0
    iters = 0
    total_tokens = 0
    correct_tokens = 0

    all_examples = []

    with torch.no_grad():
        for batch in dataloader:
            src, tgt, src_lens, tgt_lens = batch
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_key_padding_mask = make_src_key_padding_mask(src, pad_idx=pad_idx)
            tgt_key_padding_mask = make_tgt_key_padding_mask(tgt_input, pad_idx=pad_idx)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

            logits = model(
                src,
                tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
            )

            logits_flat = logits.view(-1, logits.size(-1))
            tgt_out_flat = tgt_output.contiguous().view(-1)
            loss = criterion(logits_flat, tgt_out_flat)

            total_loss += loss.item()
            iters += 1

            # Greedy decoding for accuracy / sample printing
            batch_size = src.size(0)
            memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
            ys = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=device)  # <sos>=1

            for _ in range(max_decoding_len):
                tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device)
                out = model.decode(
                    ys,
                    memory,
                    tgt_key_padding_mask=(ys == pad_idx),
                    memory_key_padding_mask=src_key_padding_mask,
                    tgt_mask=tgt_mask,
                )
                next_logits = out[:, -1, :]
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tokens], dim=1)

            # ys includes <sos> + tokens. Convert to words and compare to tgt
            predicted_seq = ys[:, 1 : 1 + tgt_output.size(1)]  # (batch, tgt_len)
            # token-level accuracy (ignoring pad)
            mask = (tgt_output != pad_idx)
            total_tokens += mask.sum().item()
            correct_tokens += ((predicted_seq == tgt_output) & mask).sum().item()

            # Save a few example strings for printing
            if src_vocab and tgt_vocab and len(all_examples) < 8:
                for i in range(min(2, batch_size)):
                    src_tokens = src[i].cpu().tolist()
                    tgt_tokens = tgt_output[i].cpu().tolist()
                    pred_tokens = predicted_seq[i].cpu().tolist()
                    src_str = src_vocab.decode(src_tokens)
                    tgt_str = tgt_vocab.decode(tgt_tokens)
                    pred_str = tgt_vocab.decode(pred_tokens)
                    all_examples.append((src_str, tgt_str, pred_str))

    avg_loss = total_loss / max(1, iters)
    token_acc = correct_tokens / (total_tokens + 1e-12)
    return avg_loss, token_acc, all_examples


# -------------------------
# Training harness / main
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic flags (might slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders_from_config(config: dict):
    preproc = DataPreprocessor(config)
    # This will read files from the config path and apply sampling if needed
    train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab = preproc.prepare_data()
    return train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab


def save_checkpoint(state: dict, save_dir: str, epoch: int):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def main(config: dict):
    # config is a dict (loaded from YAML or provided programmatically)
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build data loaders and vocabularies (uses your preprocessor which builds vocab from training data)
    train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab = build_dataloaders_from_config(config)

    src_vocab_size = len(roman_vocab)
    tgt_vocab_size = len(devanagari_vocab)
    pad_idx = roman_vocab.char2idx.get("<pad>", 0)
    sos_idx = roman_vocab.char2idx.get("<sos>", 1)
    eos_idx = roman_vocab.char2idx.get("<eos>", 2)

    # Model selection (only transformer implemented here, but config has model_type option)
    model_type = config.get("model_type", "transformer")
    print(model_type)
    if model_type != "transformer":
        raise NotImplementedError("Only transformer model_type is implemented in this script.")

    d_model = config.get("d_model", 256)
    print(f'd_model')

    nhead = config.get("nhead", 8)
    num_encoder_layers=config.get("num_encoder_layers", 2)
    num_decoder_layers=config.get("num_decoder_layers", 2)
    dim_feedforward=config.get("dim_feedforward", 512)

    print(f'nhead: {nhead} \n enco layers: {num_encoder_layers} \n deco layers: {num_decoder_layers} \n dim_feedforwar: {dim_feedforward}')

    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_decoder_layers=config.get("num_decoder_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("transformer_dropout", 0.1),
        max_len=config.get("max_len", 5000),
        pad_idx=pad_idx,
    ).to(device)

    min_lr = config.get("min_lr", 1e-6)
    if isinstance(min_lr, str):
        min_lr = float(min_lr)

    # Optimizer / scheduler / loss
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.get("lr_decay_factor", 0.8),
        patience=config.get("lr_patience", 3),
        min_lr=min_lr,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    # model_dim = config.get("d_model", 256)
    # warmup_steps = 4000
    # def noam_lr_lambda(step):
    #     step = max(1, step)
    #     return (model_dim ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lr_lambda)


    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    num_epochs = config.get("num_epochs", 50)
    grad_clip = config.get("grad_clip", 1.0)
    teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.99)
    final_teacher_forcing = config.get("final_teacher_forcing", 0.2)
    teacher_forcing_decay = config.get("teacher_forcing_decay", 0.95)

    # Early stopping
    best_val = float("inf")
    epochs_no_improve = 0
    patience = config.get("patience", 7)
    save_dir = config.get("save_dir", "./ckpts")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{num_epochs} - teacher_forcing={teacher_forcing_ratio:.4f}")

        train_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            pad_idx,
            grad_clip,
            teacher_forcing_ratio,
            scheduler
        )
        
        t1 = time.time()
        
        # Evaluate every 3 epochs OR on first epoch OR on last epoch
        if epoch % 5 == 0 or epoch == num_epochs:
            val_loss, val_acc, examples = evaluate_one_epoch(
                model,
                criterion,
                val_loader,
                device,
                pad_idx,
                max_decoding_len=200,
                src_vocab=roman_vocab,
                tgt_vocab=devanagari_vocab,
            )
            
            # Update scheduler based on validation loss
            # scheduler.step(val_loss)
            
            print(f"Epoch {epoch} done in {t1-t0:.1f}s — train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_token_acc={val_acc:.4f}")
            
            # Print examples only when we do full evaluation
            print("Examples (src | gold | pred):")
            for s, g, p in examples[:6]:
                print(f"  {s}  |  {g}  |  {p}")
                
            # Save checkpoint on improvement (only when we evaluate)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")
                
            save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                        "roman_vocab": roman_vocab,
                        "devanagari_vocab": devanagari_vocab,
                    },
                    save_dir,
                    epoch,
                )
                
        else:
            # For epochs without evaluation, just print training info
            print(f"Epoch {epoch} done in {t1-t0:.1f}s — train_loss={train_loss:.4f} (no eval this epoch)")
            
            # For non-evaluation epochs, we still need to track early stopping
            # But since we don't have val_loss, we increment the counter
            # epochs_no_improve += 1
            print(f"No evaluation this epoch - early stopping counter: {epochs_no_improve}")

        # Early stop (check every epoch, but only reset on evaluation epochs)
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        # Decay teacher forcing every epoch
        teacher_forcing_ratio = max(final_teacher_forcing, teacher_forcing_ratio * teacher_forcing_decay)

    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_loss, test_acc, test_examples = evaluate_one_epoch(
        model,
        criterion,
        test_loader,
        device,
        pad_idx,
        max_decoding_len=200,
        src_vocab=roman_vocab,
        tgt_vocab=devanagari_vocab,
    )
    print(f"Test loss: {test_loss:.4f}, Test token accuracy: {test_acc:.4f}")
    print("Test examples:")
    for s, g, p in test_examples[:10]:
        print(f"  {s}  |  {g}  |  {p}")


# -------------------------
# CLI: read YAML config and run
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
