import json
import torch
from collections import defaultdict
import torch.nn.functional as F
import argparse
from train_transformer import TransformerSeq2Seq
from tqdm import tqdm
import math

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

def transliterate_word_transformer(model, roman_word, roman_vocab, devanagari_vocab, device, max_length=80):
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
            
            # Stop if EOS token
            eos_idx = devanagari_vocab.char2idx.get("<eos>", 2)
            if next_token.item() == eos_idx:
                break
                
            decoded_indices.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Safety break if sequence gets too long
            if len(decoded_indices) >= max_length - 1:
                break
        
        # Convert indices to characters
        transliterated = devanagari_vocab.decode(decoded_indices)
        return transliterated

#########################################################
# 🔹 Enhanced Evaluation Metric Functions
#########################################################

def LCS_length(s1, s2):
    """Longest Common Subsequence length"""
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
    """Calculate F1 score between candidate and best matching reference"""
    if not candidate or not references:
        return 0.0, ''
    
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, best_ref)
    
    # Find reference with best LCS match
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2 * lcs) < (len(best_ref) - 2 * best_ref_lcs):
            best_ref, best_ref_lcs = ref, lcs
    
    precision = best_ref_lcs / len(candidate) if candidate else 0.0
    recall = best_ref_lcs / len(best_ref) if best_ref else 0.0
    
    if precision + recall == 0:
        return 0.0, best_ref
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1, best_ref

def mean_average_precision(candidates, references, n):
    """Calculate Mean Average Precision"""
    total, num_correct = 0.0, 0
    for k in range(n):
        if k < len(candidates) and candidates[k] in references:
            num_correct += 1
        total += num_correct / (k + 1)
    return total / n

def inverse_rank(candidates, reference):
    """Calculate Inverse Rank"""
    for i, cand in enumerate(candidates):
        if cand == reference:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_metrics(predictions, references):
    """Comprehensive evaluation with multiple metrics"""
    acc, f, f_best, mrr, map_ref = {}, {}, {}, {}, {}
    
    for src_word, refs in references.items():
        preds = predictions.get(src_word, [])
        if preds:
            # Word-level exact accuracy
            acc[src_word] = 1.0 if preds[0] in refs else 0.0
            
            # F1 score with best matching reference
            f[src_word], f_best[src_word] = f_score(preds[0], refs)
            
            # Ranking metrics
            mrr[src_word] = max(inverse_rank(preds, ref) for ref in refs)
            map_ref[src_word] = mean_average_precision(preds, refs, len(refs))
        else:
            acc[src_word], f[src_word], f_best[src_word] = 0.0, 0.0, ''
            mrr[src_word], map_ref[src_word] = 0.0, 0.0
    
    return acc, f, f_best, mrr, map_ref

class ComprehensiveTestEvaluator:
    def __init__(self, model, roman_vocab, devanagari_vocab, device):
        self.model = model
        self.roman_vocab = roman_vocab
        self.devanagari_vocab = devanagari_vocab
        self.device = device
        
    def run_complete_evaluation(self, test_file_path):
        """Run complete evaluation with comprehensive metrics and failure analysis"""
        
        # 1. Load test data
        test_data = self.load_test_data(test_file_path)
        print(f"📊 Loaded {len(test_data)} test examples")
        
        # 2. Generate predictions
        print("🚀 Generating predictions...")
        predictions = {}
        for roman_word, references in tqdm(test_data.items(), desc="Transliterating"):
            prediction = transliterate_word_transformer(
                self.model, roman_word, self.roman_vocab, self.devanagari_vocab, self.device
            )
            predictions[roman_word] = [prediction]
        
        # 3. Calculate comprehensive metrics
        print("📈 Calculating evaluation metrics...")
        acc, f_scores, f_best_refs, mrr_scores, map_ref_scores = evaluate_metrics(predictions, test_data)
        
        N = len(acc)
        acc_score = sum(acc.values()) / N
        f_score_val = sum(f_scores.values()) / N
        mrr_score = sum(mrr_scores.values()) / N
        map_ref_score = sum(map_ref_scores.values()) / N
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Word-level Exact Accuracy (ACC):          {acc_score:.4f}")
        print(f"Character-level F1 Score (Mean F-score):  {f_score_val:.4f}")
        print(f"Mean Reciprocal Rank (MRR):               {mrr_score:.4f}")
        print(f"Mean Average Precision (MAP_ref):         {map_ref_score:.4f}")
        print(f"\nTotal test examples: {len(test_data)}")
        print(f"Correct predictions:  {sum(acc.values())}")
        
        # 4. Analyze failures by character and orthographic units
        print("\n" + "="*60)
        print("🔍 DETAILED FAILURE ANALYSIS")
        print("="*60)
        
        char_failures, unit_failures = self.analyze_failures_by_type(predictions, test_data)
        
        print("\n📊 TOP 15 CHARACTER-LEVEL FAILURES:")
        print("-" * 40)
        for i, (char, error_rate) in enumerate(char_failures[:15], 1):
            print(f"{i:2d}. '{char}': {error_rate:.3f} error rate")
            
        print("\n📊 TOP 15 ORTHOGRAPHIC UNIT FAILURES:")
        print("-" * 45)
        for i, (unit, error_rate) in enumerate(unit_failures[:15], 1):
            print(f"{i:2d}. '{unit}': {error_rate:.3f} error rate")
        
        # 5. Additional analysis: F1 score distribution
        self.analyze_f1_distribution(f_scores)
        
        return {
            'accuracy': acc_score,
            'f1_score': f_score_val,
            'mrr': mrr_score,
            'map_ref': map_ref_score,
            'char_failures': char_failures,
            'unit_failures': unit_failures,
            'predictions': predictions
        }
    
    def analyze_failures_by_type(self, predictions, test_data):
        """Analyze failures at both character and orthographic unit level"""
        char_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
        unit_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        print("🔬 Analyzing character and orthographic unit failures...")
        
        for roman_word, refs in tqdm(test_data.items(), desc="Failure Analysis"):
            pred = predictions.get(roman_word, [''])[0]
            is_correct = any(pred == ref for ref in refs)
            
            # Analyze each reference
            for ref in refs:
                # Character-level analysis
                for char in ref:
                    char_errors[char]['total'] += 1
                    if not is_correct:
                        char_errors[char]['errors'] += 1
                
                # Orthographic unit analysis
                units = self.extract_orthographic_units(ref)
                for unit in units:
                    unit_errors[unit]['total'] += 1
                    if not is_correct:
                        unit_errors[unit]['errors'] += 1
        
        # Calculate error rates
        char_error_rates = self.calculate_error_rates(char_errors, min_count=5)
        unit_error_rates = self.calculate_error_rates(unit_errors, min_count=3)
        
        return char_error_rates, unit_error_rates
    
    def extract_orthographic_units(self, devanagari_word):
        """Enhanced orthographic unit extraction for Devanagari"""
        units = []
        i = 0
        n = len(devanagari_word)
        
        while i < n:
            current_char = devanagari_word[i]
            
            # Look ahead for modifiers and conjuncts
            j = i + 1
            while j < n:
                next_char = devanagari_word[j]
                # Check for vowel signs, anusvara, visarga, halant, nukta
                is_modifier = (
                    ('\u093E' <= next_char <= '\u094C') or  # Vowel signs
                    next_char == '\u094D' or  # Halant (virama)
                    next_char == '\u093C' or  # Nukta
                    next_char == '\u0902' or  # Anusvara
                    next_char == '\u0901' or  # Anunasika
                    next_char == '\u0903'     # Visarga
                )
                if is_modifier:
                    current_char += next_char
                    j += 1
                else:
                    break
            
            units.append(current_char)
            i = j
        
        return units
    
    def calculate_error_rates(self, error_dict, min_count=3):
        """Calculate error rates filtering by minimum count"""
        rates = []
        for item, counts in error_dict.items():
            if counts['total'] >= min_count:
                error_rate = counts['errors'] / counts['total']
                rates.append((item, error_rate))
        return sorted(rates, key=lambda x: x[1], reverse=True)
    
    def analyze_f1_distribution(self, f_scores):
        """Analyze distribution of F1 scores"""
        f1_values = list(f_scores.values())
        
        if not f1_values:
            return
            
        perfect_f1 = sum(1 for f1 in f1_values if f1 == 1.0)
        good_f1 = sum(1 for f1 in f1_values if 0.7 <= f1 < 1.0)
        medium_f1 = sum(1 for f1 in f1_values if 0.4 <= f1 < 0.7)
        poor_f1 = sum(1 for f1 in f1_values if f1 < 0.4)
        
        total = len(f1_values)
        
        print(f"\n📊 F1 SCORE DISTRIBUTION:")
        print(f"   Perfect (F1 = 1.0):    {perfect_f1:4d} ({perfect_f1/total:.1%})")
        print(f"   Good (0.7 ≤ F1 < 1.0): {good_f1:4d} ({good_f1/total:.1%})")
        print(f"   Medium (0.4 ≤ F1 < 0.7): {medium_f1:4d} ({medium_f1/total:.1%})")
        print(f"   Poor (F1 < 0.4):       {poor_f1:4d} ({poor_f1/total:.1%})")
    
    def load_test_data(self, test_file_path):
        """Load test data with multiple references per word"""
        test_data = {}
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    roman = obj.get("english word", "").strip().lower()
                    dev = obj.get("native word", "").strip()
                    if roman and dev:
                        if roman not in test_data:
                            test_data[roman] = []
                        test_data[roman].append(dev)
                except json.JSONDecodeError:
                    continue
        
        # Print some statistics
        multi_ref_count = sum(1 for refs in test_data.values() if len(refs) > 1)
        print(f"   • Words with multiple references: {multi_ref_count}")
        
        return test_data

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Transformer Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test data JSONL")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load model and vocabularies
        print("📥 Loading model checkpoint...")
        checkpoint = quick_load(args.checkpoint, device)
        model, roman_vocab, devanagari_vocab = initialize_model_from_checkpoint(checkpoint, device)
        
        # Run evaluation
        evaluator = ComprehensiveTestEvaluator(model, roman_vocab, devanagari_vocab, device)
        results = evaluator.run_complete_evaluation(args.test_file)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            # Convert to JSON-serializable format
            serializable_results = {
                'metrics': {
                    'accuracy': results['accuracy'],
                    'f1_score': results['f1_score'],
                    'mrr': results['mrr'],
                    'map_ref': results['map_ref']
                },
                'top_char_failures': results['char_failures'][:50],
                'top_unit_failures': results['unit_failures'][:150]
            }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {args.output}")
        
        # Print sampling insights
        print("\n" + "="*60)
        print("🎯 SAMPLING STRATEGY INSIGHTS")
        print("="*60)
        print("Focus your next sampling iteration on these problematic units:")
        for i, (unit, error_rate) in enumerate(results['unit_failures'][:10], 1):
            print(f"   {i}. '{unit}' (error rate: {error_rate:.1%})")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()