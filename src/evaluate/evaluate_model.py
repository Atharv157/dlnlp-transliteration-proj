import torch
import json
import codecs
from xml.dom.minidom import Document
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.lstm_seq2seq import initialize_lstm_model
from data.vocabulary import RomanVocabulary, DevanagariVocabulary

def quick_load(checkpoint_path, device):
    """Quick load with weights_only=False - USE ONLY IF YOU TRUST THE SOURCE"""
    print("⚠️  Using unsafe loading - only use with trusted checkpoints!")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def print_vocabularies(roman_vocab, devanagari_vocab):
    """Print vocabulary information"""
    print("\n" + "="*50)
    print("📖 VOCABULARY INFORMATION")
    print("="*50)
    
    print(f"\n🔤 Roman Vocabulary ({len(roman_vocab)} characters):")
    print("Special tokens:")
    for idx in range(4):  # <pad>, <sos>, <eos>, <unk>
        if idx in roman_vocab.idx2char:
            print(f"  {idx}: '{roman_vocab.idx2char[idx]}'")
    
    print("\nRegular characters:")
    regular_chars = []
    for idx, char in roman_vocab.idx2char.items():
        if idx >= 4:  # Skip special tokens
            regular_chars.append(char)
    print(f"  {', '.join(sorted(regular_chars))}")
    
    print(f"\n📝 Devanagari Vocabulary ({len(devanagari_vocab)} characters):")
    print("Special tokens:")
    for idx in range(4):  # <pad>, <sos>, <eos>, <unk>
        if idx in devanagari_vocab.idx2char:
            print(f"  {idx}: '{devanagari_vocab.idx2char[idx]}'")
    
    print("\nDevanagari characters:")
    devanagari_chars = []
    for idx, char in devanagari_vocab.idx2char.items():
        if idx >= 4:  # Skip special tokens
            devanagari_chars.append(char)
    
    # Print in groups for better readability
    for i in range(0, len(devanagari_chars), 10):
        print(f"  {', '.join(devanagari_chars[i:i+10])}")

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

def test_transliteration(model, roman_vocab, devanagari_vocab, device):
    """Test the model on some sample words"""
    print("\n" + "="*50)
    print("🧪 TRANSLITERATION TEST")
    print("="*50)
    
    test_words = [
        "namaste", "bharat", "hindi", "school", "kitab",
        "aam", "pani", "desh", "vidya", "dost",
        "computer", "mobile", "technology", "india", "language"
    ]
    
    for word in test_words:
        prediction = transliterate_word(model, word, roman_vocab, devanagari_vocab, device)
        print(f"  {word:15} → {prediction}")

def create_results_xml(test_data, model, roman_vocab, devanagari_vocab, device, output_path):
    """Create XML file in NEWS format with model predictions - ALL examples"""
    doc = Document()
    
    # Create root element
    results = doc.createElement('TransliterationTaskResults')
    results.setAttribute('SourceLang', 'Roman')
    results.setAttribute('TargetLang', 'Devanagari')
    results.setAttribute('GroupID', 'DLNLP')
    results.setAttribute('RunID', 'LSTM_Seq2Seq')
    results.setAttribute('RunType', 'automatic')
    results.setAttribute('Comments', 'LSTM with Attention Model')
    doc.appendChild(results)
    
    # Process ALL test examples (no limit)
    print(f"\n📊 Generating predictions for {len(test_data)} examples...")
    
    for i, (roman_word, devanagari_ref) in enumerate(test_data):
        name_elem = doc.createElement('Name')
        name_elem.setAttribute('ID', str(i+1))
        
        # Source name
        src_elem = doc.createElement('SourceName')
        src_text = doc.createTextNode(roman_word)
        src_elem.appendChild(src_text)
        name_elem.appendChild(src_elem)
        
        # Target name (prediction)
        tgt_elem = doc.createElement('TargetName')
        tgt_elem.setAttribute('ID', '1')
        
        # Get model prediction
        prediction = transliterate_word(model, roman_word, roman_vocab, devanagari_vocab, device)
        tgt_text = doc.createTextNode(prediction)
        tgt_elem.appendChild(tgt_text)
        name_elem.appendChild(tgt_elem)
        
        results.appendChild(name_elem)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} examples")
    
    # Write to file
    with codecs.open(output_path, 'w', 'utf-8') as f:
        f.write(doc.toprettyxml(indent='  '))
    
    print(f"✅ Predictions saved to: {output_path}")

def load_test_data(test_file):
    """Load ALL test data from JSON file - no limits"""
    print(f"Loading test data from: {test_file}")
    
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths - update these to your actual paths
    test_file = 'data/raw/hin/hin_test.json'
    checkpoint_path = 'saved_models/best_model_lstm_seq2seq.pth'
    output_xml = 'model_predictions.xml'
    
    # Quick load
    checkpoint = quick_load(checkpoint_path, device)
    
    # Extract everything
    config = checkpoint.get('config', {})
    roman_vocab = checkpoint['roman_vocab']
    devanagari_vocab = checkpoint['devanagari_vocab']
    
    print(f"✅ Roman vocabulary size: {len(roman_vocab)}")
    print(f"✅ Devanagari vocabulary size: {len(devanagari_vocab)}")
    
    # Print vocabulary details
    print_vocabularies(roman_vocab, devanagari_vocab)
    
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
    
    # Test transliteration on sample words
    test_transliteration(model, roman_vocab, devanagari_vocab, device)
    
    # Load ALL test data (no limits)
    test_data = load_test_data(test_file)
    
    # Generate predictions XML for ALL examples (no limits)
    create_results_xml(test_data, model, roman_vocab, devanagari_vocab, device, output_xml)
    
    print(f"\n🎉 Evaluation ready! Next steps:")
    print(f"1. Run: python final_evaluation.py -t test_reference.xml -i {output_xml}")
    print(f"2. Check the generated file: {output_xml}")

if __name__ == "__main__":
    main()