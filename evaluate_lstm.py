import torch
import json
import sys
import os

# Add the current directory to path
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_seq2seq import initialize_lstm_model
from src.data.vocabulary import RomanVocabulary, DevanagariVocabulary

def quick_load(checkpoint_path, device):
    """Quick load with weights_only=False - USE ONLY IF YOU TRUST THE SOURCE"""
    print("⚠️  Using unsafe loading - only use with trusted checkpoints!")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def transliterate_word(model, roman_word, roman_vocab, devanagari_vocab, device, max_length=50):
    """Transliterate a single word using the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Encode input
        roman_chars = roman_vocab.split_roman_word(roman_word)
        roman_encoded = roman_vocab.encode(roman_chars)
        src_tensor = torch.tensor(roman_encoded).unsqueeze(0).to(device)
        
        # Encode source
        encoder_outputs, encoder_hidden, encoder_cell = model.encoder(src_tensor)
        
        # Initialize decoder
        if model.encoder.lstm.bidirectional:
            num_layers = model.encoder.num_layers
            hidden_dim = model.encoder.hidden_dim
            batch_size = 1
            
            encoder_hidden_reshaped = encoder_hidden.view(
                num_layers, 2, batch_size, hidden_dim
            )
            
            last_forward = encoder_hidden_reshaped[-1, 0]
            last_backward = encoder_hidden_reshaped[-1, 1]
            
            decoder_hidden_init = torch.cat([last_forward, last_backward], dim=1)
            decoder_hidden = model.init_hidden_proj(decoder_hidden_init)
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        else:
            decoder_hidden = encoder_hidden[-1].unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        # Start with <sos> token
        decoder_input = torch.tensor([devanagari_vocab.char2idx['<sos>']]).to(device)
        
        decoded_indices = []
        
        for _ in range(max_length):
            output, decoder_hidden, decoder_cell, _ = model.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            # Get most likely character
            top1 = output.argmax(1)
            decoder_input = top1
            
            # Stop if EOS token
            if top1.item() == devanagari_vocab.char2idx['<eos>']:
                break
                
            decoded_indices.append(top1.item())
        
        # Convert indices to characters
        transliterated = devanagari_vocab.decode(decoded_indices)
        return transliterated

def load_test_data(test_file):
    """Load test data from JSON file"""
    print(f"📥 Loading test data from: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = []
        for line in f:
            try:
                item = json.loads(line.strip())
                roman_word = item.get('english word', '').strip()
                devanagari_word = item.get('native word', '').strip()
                if roman_word and devanagari_word:
                    test_data.append((roman_word, devanagari_word))
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Loaded {len(test_data)} test examples")
    return test_data

def LCS_length(s1, s2):
    """Calculate the length of the longest common subsequence"""
    m = len(s1)
    n = len(s2)
    C = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C[m][n]

def calculate_metrics(predictions, references):
    """Calculate all evaluation metrics directly"""
    print("\n📊 Calculating evaluation metrics...")
    
    total_acc = 0
    total_f1 = 0
    total_mrr = 0
    total_map_ref = 0
    total_exact_matches = 0
    
    print("🔍 Sample predictions:")
    for i, (pred, ref_data) in enumerate(zip(predictions, references)):
        # ref_data is a tuple: (roman_word, devanagari_reference)
        roman_word, reference = ref_data
        
        # 1. Accuracy (Exact Match)
        exact_match = 1 if pred == reference else 0
        total_acc += exact_match
        total_exact_matches += exact_match
        
        # 2. F1 Score (Character-level)
        lcs_len = LCS_length(pred, reference)
        if len(pred) > 0 and len(reference) > 0:
            precision = lcs_len / len(pred)
            recall = lcs_len / len(reference)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
        else:
            f1 = 0
        total_f1 += f1
        
        # 3. MRR (Mean Reciprocal Rank) - simplified for single candidate
        mrr = 1.0 if pred == reference else 0.0
        total_mrr += mrr
        
        # 4. MAP_ref (Mean Average Precision) - simplified for single reference
        map_ref = 1.0 if pred == reference else 0.0
        total_map_ref += map_ref
        
        # Print some examples
        if i < 5:  # Show first 5 examples
            match_indicator = "✅" if exact_match else "❌"
            print(f"  {match_indicator} '{roman_word}' → '{pred}' (ref: '{reference}')")
    
    n = len(predictions)
    
    # Calculate averages
    accuracy = total_acc / n
    avg_f1 = total_f1 / n
    avg_mrr = total_mrr / n
    avg_map_ref = total_map_ref / n
    
    return {
        'accuracy': accuracy,
        'f1_score': avg_f1,
        'mrr': avg_mrr,
        'map_ref': avg_map_ref,
        'exact_matches': total_exact_matches,
        'total_examples': n
    }

def generate_predictions(model, test_data, roman_vocab, devanagari_vocab, device):
    """Generate predictions for all test data"""
    print(f"\n🔮 Generating predictions for {len(test_data)} examples...")
    
    predictions = []
    
    for i, (roman_word, devanagari_ref) in enumerate(test_data):
        # Get model prediction
        prediction = transliterate_word(model, roman_word, roman_vocab, devanagari_vocab, device)
        predictions.append(prediction)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} examples")
    
    return predictions

def print_detailed_results(metrics, predictions, test_data):
    """Print comprehensive results"""
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📈 METRICS SUMMARY:")
    print("-" * 40)
    print(f"Word-level Accuracy (Exact Match): {metrics['accuracy']:.4f} ({metrics['exact_matches']}/{metrics['total_examples']})")
    print(f"Character-level F1 Score:          {metrics['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR):        {metrics['mrr']:.4f}")
    print(f"Mean Average Precision (MAP_ref):  {metrics['map_ref']:.4f}")
    
    print(f"\n📋 TABULAR COMPARISON:")
    print("=" * 50)
    print(f"{'Metric':<35} {'Score':<10} {'Details':<15}")
    print("-" * 50)
    print(f"{'Word-level Accuracy':<35} {metrics['accuracy']:.4f}   {metrics['exact_matches']}/{metrics['total_examples']}")
    print(f"{'Character-level F1':<35} {metrics['f1_score']:.4f}")
    print(f"{'MRR':<35} {metrics['mrr']:.4f}")
    print(f"{'MAP_ref':<35} {metrics['map_ref']:.4f}")
    print("-" * 50)
    
    # Show some examples of correct and incorrect predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print("-" * 60)
    
    correct_examples = []
    incorrect_examples = []
    
    for i, (pred, (roman, ref)) in enumerate(zip(predictions, test_data)):
        if pred == ref:
            correct_examples.append((roman, pred, ref))
        else:
            incorrect_examples.append((roman, pred, ref))
    
    print(f"✅ Correct predictions (showing 3):")
    for i, (roman, pred, ref) in enumerate(correct_examples[:3]):
        print(f"   {roman:15} → {pred:15} (ref: {ref})")
    
    if incorrect_examples:
        print(f"❌ Incorrect predictions (showing 3):")
        for i, (roman, pred, ref) in enumerate(incorrect_examples[:3]):
            print(f"   {roman:15} → {pred:15} (ref: {ref})")
    else:
        print(f"   🎉 All predictions are correct!")
    
    print(f"\n📊 PREDICTION DISTRIBUTION:")
    print(f"   Total examples:    {metrics['total_examples']}")
    print(f"   Correct:           {metrics['exact_matches']} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Incorrect:         {metrics['total_examples'] - metrics['exact_matches']} ({(1-metrics['accuracy'])*100:.1f}%)")

def main():
    """Main function - direct evaluation without XML"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Configuration
    test_file = 'data/raw/hin/hin_test.json'  # Your test JSON file
    checkpoint_path = 'saved_models/lstm/checkpoint_expt1_epoch_17.pth'
    
    # Step 1: Load model and vocabularies
    print("\n" + "="*60)
    print("📥 LOADING MODEL AND VOCABULARIES")
    print("="*60)
    
    checkpoint = quick_load(checkpoint_path, device)
    config = checkpoint.get('config', {})
    roman_vocab = checkpoint['roman_vocab']
    devanagari_vocab = checkpoint['devanagari_vocab']
    
    print(f"✅ Roman vocabulary size: {len(roman_vocab)}")
    print(f"✅ Devanagari vocabulary size: {len(devanagari_vocab)}")
    
    # Initialize model
    model = initialize_lstm_model(
        roman_vocab_size=len(roman_vocab),
        devanagari_vocab_size=len(devanagari_vocab),
        device=device,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Step 2: Load test data
    test_data = load_test_data(test_file)
    
    # Step 3: Generate predictions
    print("\n" + "="*60)
    print("🔮 GENERATING PREDICTIONS")
    print("="*60)
    
    predictions = generate_predictions(model, test_data, roman_vocab, devanagari_vocab, device)
    
    # Step 4: Calculate metrics
    print("\n" + "="*60)
    print("📊 CALCULATING METRICS")
    print("="*60)
    
    # Pass test_data as references since it contains both roman words and devanagari references
    metrics = calculate_metrics(predictions, test_data)
    
    # Step 5: Print results
    print_detailed_results(metrics, predictions, test_data)
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"💡 No temporary files created - everything processed in memory!")

if __name__ == "__main__":
    main()