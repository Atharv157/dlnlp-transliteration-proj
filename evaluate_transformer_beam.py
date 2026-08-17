#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import json
import argparse
import os
import math
from tqdm import tqdm

from train_transformer import TransformerSeq2Seq

##########################################################
# 🔹 Transformer Loading and Inference Functions
##########################################################

def quick_load(checkpoint_path, device):
    """Quick load with weights_only=False - USE ONLY IF YOU TRUST THE SOURCE"""
    print("⚠️  Using unsafe loading - only use with trusted checkpoints!")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def initialize_model_from_checkpoint(checkpoint, device):
    """Initialize model from checkpoint"""
    config = checkpoint.get('config', {})
    roman_vocab = checkpoint['roman_vocab']
    devanagari_vocab = checkpoint['devanagari_vocab']
    
    # Get model parameters from config or use defaults
    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 8)
    num_encoder_layers = config.get("num_encoder_layers", 2)
    num_decoder_layers = config.get("num_decoder_layers", 2)
    dim_feedforward = config.get("dim_feedforward", 512)
    dropout = config.get("transformer_dropout", 0.1)
    max_len = config.get("max_len", 5000)
    
    pad_idx = roman_vocab.char2idx.get("<pad>", 0)
    
    # Initialize model
    model = TransformerSeq2Seq(
        src_vocab_size=len(roman_vocab),
        tgt_vocab_size=len(devanagari_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_idx=pad_idx,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, roman_vocab, devanagari_vocab


def make_src_key_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src == pad_idx)


def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


def transliterate_word_transformer(model, roman_word, roman_vocab, devanagari_vocab, device, max_length=50):
    """Transliterate using Transformer with proper decoding"""
    model.eval()
    
    with torch.no_grad():
        # Prepare source
        roman_chars = roman_vocab.split_roman_word(roman_word)
        roman_encoded = roman_vocab.encode(roman_chars)
        src_tensor = torch.tensor(roman_encoded).unsqueeze(0).to(device)
        
        # Create source mask
        src_key_padding_mask = make_src_key_padding_mask(src_tensor, pad_idx=roman_vocab.char2idx.get("<pad>", 0))
        
        # Encode source
        memory = model.encode(src_tensor, src_key_padding_mask=src_key_padding_mask)
        
        # Start with SOS token
        sos_idx = devanagari_vocab.char2idx.get("<sos>", 1)
        decoder_input = torch.tensor([[sos_idx]], device=device)
        
        decoded_indices = []
        
        for step in range(max_length):
            # Create target mask and padding mask
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), device=device)
            tgt_key_padding_mask = make_src_key_padding_mask(decoder_input, pad_idx=devanagari_vocab.char2idx.get("<pad>", 0))
            
            # Transformer decoder
            output_step = model.decode(
                decoder_input,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            # Get prediction for the last token
            next_token_logits = output_step[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Debug step
            # if step < 3:  # Print first few steps for debugging
            #     print(f"    Step {step}: decoder_input={decoder_input.cpu().numpy()}, next_token={next_token.item()}")
            #     print(f"    Output shape: {output_step.shape}, max_prob: {torch.softmax(next_token_logits[0], dim=0).max().item():.3f}")
            
            # Stop if EOS token
            eos_idx = devanagari_vocab.char2idx.get("<eos>", 2)
            if next_token.item() == eos_idx:
                break
                
            decoded_indices.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Safety break if sequence gets too long
            if len(decoded_indices) >= max_length - 1:
                break
        
        # print(f"  Final decoded indices: {decoded_indices}")
        # print(f"  Decoded chars: {[devanagari_vocab.idx2char.get(idx, '?') for idx in decoded_indices]}")
        
        # Convert indices to characters
        transliterated = devanagari_vocab.decode(decoded_indices)
        # print(f"  Final transliteration: '{transliterated}'")
        
        return transliterated


##########################################################
# 🔹 Evaluation Metric Functions (from NEWS)
##########################################################

def LCS_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def f_score(candidate, references):
    if not candidate or not references:
        return 0.0, ''
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, best_ref)
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2 * lcs) < (len(best_ref) - 2 * best_ref_lcs):
            best_ref, best_ref_lcs = ref, lcs
    precision = best_ref_lcs / len(candidate) if candidate else 0.0
    recall = best_ref_lcs / len(best_ref) if best_ref else 0.0
    if precision + recall == 0:
        return 0.0, best_ref
    return 2 * precision * recall / (precision + recall), best_ref


def mean_average_precision(candidates, references, n):
    total, num_correct = 0.0, 0
    for k in range(n):
        if k < len(candidates) and candidates[k] in references:
            num_correct += 1
        total += num_correct / (k + 1)
    return total / n


def inverse_rank(candidates, reference):
    for i, cand in enumerate(candidates):
        if cand == reference:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(predictions, references):
    acc, f, f_best, mrr, map_ref = {}, {}, {}, {}, {}
    for src_word, refs in references.items():
        preds = predictions.get(src_word, [])
        if preds:
            acc[src_word] = 1.0 if preds[0] in refs else 0.0
            f[src_word], f_best[src_word] = f_score(preds[0], refs)
            mrr[src_word] = max(inverse_rank(preds, ref) for ref in refs)
            map_ref[src_word] = mean_average_precision(preds, refs, len(refs))
        else:
            acc[src_word], f[src_word], f_best[src_word] = 0.0, 0.0, ''
            mrr[src_word], map_ref[src_word] = 0.0, 0.0
    return acc, f, f_best, mrr, map_ref

##########################################################
# 🔹 Data Loading and Prediction
##########################################################

def load_json_test_data(test_file):
    print(f"📖 Loading test data from: {test_file}")
    data = {}
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                roman = obj.get("english word", "").strip()
                dev = obj.get("native word", "").strip()
                if roman and dev:
                    data[roman] = [dev]
            except json.JSONDecodeError:
                continue
    print(f"✅ Loaded {len(data)} examples.")
    return data


def generate_predictions(model, roman_vocab, dev_vocab, device, test_data, use_beam=True, beam_width=5):
    print("\n🚀 Generating predictions...")
    predictions = {}
    for i, roman_word in tqdm(enumerate(test_data.keys())):
        try:
            if use_beam:
                pred = transliterate_word_beam_search(model, roman_word.lower(), roman_vocab, dev_vocab, device, beam_width=beam_width)
            else:
                pred = transliterate_word_transformer(model, roman_word.lower(), roman_vocab, dev_vocab, device)
            predictions[roman_word] = [pred.upper()]
        except Exception as e:
            predictions[roman_word] = [""]
            print(f"⚠️ Error on '{roman_word}': {e}")
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")
    return predictions



def check_vocabularies(roman_vocab, devanagari_vocab):
    """Check if vocabularies are properly loaded"""
    print("\n" + "="*50)
    print("🔍 VOCABULARY CHECK")
    print("="*50)
    
    print(f"Roman vocab size: {len(roman_vocab)}")
    print(f"Devanagari vocab size: {len(devanagari_vocab)}")
    
    # Check special tokens
    special_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
    for token in special_tokens:
        roman_idx = roman_vocab.char2idx.get(token)
        dev_idx = devanagari_vocab.char2idx.get(token)
        print(f"  {token}: roman={roman_idx}, devanagari={dev_idx}")
    
    # Check some sample mappings
    print("\nSample roman mappings:")
    for char in ['a', 'm', 'p', 'n']:
        if char in roman_vocab.char2idx:
            print(f"  '{char}' -> {roman_vocab.char2idx[char]}")
    
    print("\nSample devanagari mappings:")
    for idx in list(devanagari_vocab.idx2char.keys())[:10]:
        char = devanagari_vocab.idx2char.get(idx, '?')
        print(f"  {idx} -> '{char}'")

def transliterate_word_beam_search(model, roman_word, roman_vocab, devanagari_vocab, device, 
                                   beam_width=5, max_length=50, length_penalty=1.0):
    """Transliterate using beam search decoding"""
    model.eval()
    with torch.no_grad():
        # Encode source sequence
        roman_chars = roman_vocab.split_roman_word(roman_word)
        roman_encoded = roman_vocab.encode(roman_chars)
        src_tensor = torch.tensor(roman_encoded).unsqueeze(0).to(device)
        src_key_padding_mask = make_src_key_padding_mask(src_tensor, pad_idx=roman_vocab.char2idx.get("<pad>", 0))
        memory = model.encode(src_tensor, src_key_padding_mask=src_key_padding_mask)

        sos_idx = devanagari_vocab.char2idx.get("<sos>", 1)
        eos_idx = devanagari_vocab.char2idx.get("<eos>", 2)
        pad_idx = devanagari_vocab.char2idx.get("<pad>", 0)

        # Each beam: (sequence tensor, log probability)
        beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]
        completed = []

        for step in range(max_length):
            new_beams = []
            for seq, score in beams:
                # Stop expanding if EOS found
                if seq[0, -1].item() == eos_idx:
                    completed.append((seq, score))
                    continue

                tgt_mask = generate_square_subsequent_mask(seq.size(1), device=device)
                tgt_key_padding_mask = make_src_key_padding_mask(seq, pad_idx=pad_idx)
                output = model.decode(
                    seq,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                    tgt_mask=tgt_mask
                )

                next_logits = output[:, -1, :]  # [1, vocab]
                log_probs = F.log_softmax(next_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                for k in range(beam_width):
                    next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)  # shape [1,1]
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, k].item()
                    new_beams.append((new_seq, new_score))

            # Keep only top-k beams
            beams = sorted(new_beams, key=lambda x: x[1] / ((len(x[0][0]) ** length_penalty)), reverse=True)[:beam_width]

            # Stop early if all beams ended with EOS
            if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        # Sort completed sequences by normalized log prob
        completed = sorted(completed, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
        best_seq = completed[0][0][0].tolist()

        # Remove SOS/EOS
        decoded_indices = [idx for idx in best_seq if idx not in (sos_idx, eos_idx, pad_idx)]
        transliterated = devanagari_vocab.decode(decoded_indices)
        return transliterated

##########################################################
# 🔹 Main Entry
##########################################################

def main():
    parser = argparse.ArgumentParser(description="Unified Transformer transliteration + evaluation (no XML)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=70)
    parser.add_argument("--beam-width", type=int, default=5, help="Beam size for beam search decoding")
    parser.add_argument("--no-beam", action="store_true", help="Disable beam search (use greedy decoding)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = quick_load(args.checkpoint, device)
    model, roman_vocab, dev_vocab = initialize_model_from_checkpoint(checkpoint, device)
    check_vocabularies(roman_vocab, dev_vocab)
    model.eval()

    test_data = load_json_test_data(args.test_file)
    preds = generate_predictions(
                model, roman_vocab, dev_vocab, device, test_data,
                use_beam=not args.no_beam, beam_width=args.beam_width
            )

    acc, f, f_best, mrr, map_ref = evaluate(preds, test_data)
    N = len(acc)
    acc_score = sum(acc.values()) / N
    f_score_val = sum(f.values()) / N
    mrr_score = sum(mrr.values()) / N
    map_ref_score = sum(map_ref.values()) / N

    print("\n📊 FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Word-level Exact Accuracy (ACC):          {acc_score:.4f}")
    print(f"Character-level F1 Score (Mean F-score):  {f_score_val:.4f}")
    print(f"MRR:                                      {mrr_score:.4f}")
    print(f"MAP_ref:                                  {map_ref_score:.4f}")
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()
