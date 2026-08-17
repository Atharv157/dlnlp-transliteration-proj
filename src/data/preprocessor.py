import json
import os
from src.data.vocabulary import RomanVocabulary, DevanagariVocabulary
from src.data.dataset import TransliterationDataset, collate_fn
from src.data.sampler import HybridSampler
from torch.utils.data import DataLoader
from pathlib import Path

# Load failure analysis from your JSON results file
def load_failure_analysis(json_file_path):
    """Load failure analysis from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        failure_data = json.load(f)
    
    # Extract the relevant parts
    failure_analysis = {
        'top_char_failures': failure_data.get('top_char_failures', []),
        'top_unit_failures': failure_data.get('top_unit_failures', [])
    }
    
    print(f"✅ Loaded failure analysis from: {json_file_path}")
    print(f"   • {len(failure_analysis['top_char_failures'])} character failures")
    print(f"   • {len(failure_analysis['top_unit_failures'])} unit failures")
    
    return failure_analysis

# import json
# import os
# from pathlib import Path
# from .vocabulary import RomanVocabulary, DevanagariVocabulary
# from .dataset import TransliterationDataset, collate_fn
# from .sampler import HybridSampler
# from torch.utils.data import DataLoader

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        seed = self.config.get('seed', 42)
        self.sampler = HybridSampler(seed=seed)
        # failure_analysis_path = 'evaluation_results.json'
        # failure_analysis = load_failure_analysis(failure_analysis_path)
        # self.sampler = HybridSampler(seed=seed, failure_analysis=failure_analysis)
    
    def load_jsonl_data(self, file_path):
        """Load data from JSONL file (Aksharantar format)"""
        data_pairs = []
        
        # Construct full path
        data_dir = self.config.get('data_dir', 'data/raw/aksharantar')
        full_path = os.path.join(data_dir, file_path)
        
        print(f"Loading data from: {full_path}")
        
        if not os.path.exists(full_path):
            print(f"❌ File not found: {full_path}")
            return []
        
        with open(full_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        # Aksharantar format: "english_word" and "native_word"
                        roman_word = item.get('english word', '').strip()
                        devanagari_word = item.get('native word', '').strip()
                        
                        if roman_word and devanagari_word:
                            data_pairs.append((roman_word, devanagari_word))
                    
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ Unexpected error on line {line_num}: {e}")
                        continue
        
        print(f"✅ Loaded {len(data_pairs)} valid examples from {file_path}")
        return data_pairs
    
    def build_vocabularies(self, data_pairs):
        """Build Roman and Devanagari vocabularies from data"""
        roman_vocab = RomanVocabulary()
        devanagari_vocab = DevanagariVocabulary()
        
        for roman_word, devanagari_word in data_pairs:
            roman_chars = roman_vocab.split_roman_word(roman_word)
            devanagari_chars = devanagari_vocab.split_devanagari_word(devanagari_word)
            
            roman_vocab.add_text(roman_chars)
            devanagari_vocab.add_text(devanagari_chars)
        
        print(f"📖 Roman vocabulary size: {len(roman_vocab)}")
        print(f"📖 Devanagari vocabulary size: {len(devanagari_vocab)}")
        
        return roman_vocab, devanagari_vocab
    
    def prepare_data(self):
        """Main method to prepare all data with separate train/val/test files"""
        # Load TRAINING data
        train_data_path = self.config.get('train_data_path')
        train_pairs = self.load_jsonl_data(train_data_path)
        
        if not train_pairs:
            raise ValueError(f"❌ No training data found at {train_data_path}")
        
        # Load VALIDATION data (separate file)
        val_data_path = self.config.get('val_data_path')
        val_pairs = self.load_jsonl_data(val_data_path)
        
        if not val_pairs:
            raise ValueError(f"❌ No validation data found at {val_data_path}")
        
        # Load TEST data (completely separate)
        test_data_path = self.config.get('test_data_path')
        test_pairs = self.load_jsonl_data(test_data_path)
        
        if not test_pairs:
            print(f"⚠️  No test data found at {test_data_path}")
            test_pairs = []
        
        print(f"📊 Data Statistics:")
        print(f"   • Training data: {len(train_pairs)} examples")
        print(f"   • Validation data: {len(val_pairs)} examples") 
        print(f"   • Test data: {len(test_pairs)} examples")
        
        # Apply hybrid sampling only to TRAINING data
        sample_size = self.config.get('sample_size', 100000)
        if sample_size < len(train_pairs):
            print(f"🎯 Sampling {sample_size} examples from {len(train_pairs)} training examples")
            sampled_train_pairs = self.sampler.hybrid_sampling(train_pairs, 
                                                               sample_size,
                                                               method=self.config.get('sampling_method', 'fast') )
            # sampled_train_pairs = self.sampler.hybrid_sampling(train_pairs, 
            #                                                    sample_size,
            #                                                    method="enhanced")
        else:
            print(f"🎯 Using all {len(train_pairs)} training examples")
            sampled_train_pairs = train_pairs
        
        # Build vocabularies from SAMPLED TRAINING data only
        print("\n📚 Building vocabularies from training data...")
        roman_vocab, devanagari_vocab = self.build_vocabularies(sampled_train_pairs)
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = self._create_data_loader(
            sampled_train_pairs, 
            roman_vocab, devanagari_vocab, batch_size, shuffle=True,
            num_workers=4,           # Add this
            pin_memory=True,         # Add this
            persistent_workers=True, # Add this
            prefetch_factor=2
            )
        val_loader = self._create_data_loader(val_pairs, roman_vocab, devanagari_vocab, batch_size, shuffle=False)
        test_loader = self._create_data_loader(test_pairs, roman_vocab, devanagari_vocab, batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab
    
    def _create_data_loader(self, data_pairs, roman_vocab, devanagari_vocab, batch_size, shuffle=False, **kwargs):
        """Create a DataLoader for given data pairs"""
        dataset = TransliterationDataset(data_pairs, roman_vocab, devanagari_vocab)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)

# Updated test function
def test_preprocessor():
    config = {
        'train_data_path': 'data/raw/train_hindi.jsonl',
        'test_data_path': 'data/raw/test_hindi.jsonl',
        'sample_size': 1000,
        'batch_size': 2,
        'train_ratio': 0.9,  # 90% of sampled data for training
        'val_ratio': 0.1,    # 10% for validation
        'seed': 42
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Create mock training data
    mock_train_data = [
        {"unique_identifier": "hin1", "native word": "जन्मदिवस", "english word": "janamdivas", "source": "Dakshina", "score": None},
        {"unique_identifier": "hin2", "native word": "रक्खा", "english word": "rakha", "source": "Dakshina", "score": None},
        {"unique_identifier": "hin3", "native word": "मिलीजुली", "english word": "milijuli", "source": "Dakshina", "score": None},
    ] * 100  # 300 training examples
    
    # Create mock test data (completely different!)
    mock_test_data = [
        {"unique_identifier": "test1", "native word": "स्कूल", "english word": "school", "source": "Dakshina", "score": None},
        {"unique_identifier": "test2", "native word": "किताब", "english word": "kitab", "source": "Dakshina", "score": None},
        {"unique_identifier": "test3", "native word": "दुनिया", "english word": "duniya", "source": "Dakshina", "score": None},
    ] * 10  # 30 test examples
    
    # Create mock files
    os.makedirs('data/raw', exist_ok=True)
    
    with open('data/raw/train_hindi.jsonl', 'w', encoding='utf-8') as f:
        for item in mock_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open('data/raw/test_hindi.jsonl', 'w', encoding='utf-8') as f:
        for item in mock_test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    preprocessor = DataPreprocessor(config)
    
    try:
        train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab = preprocessor.prepare_data()
        print("Data pipeline test successful!")
        
        # Test one batch from each
        print("\nTraining batch:")
        for roman_batch, dev_batch in train_loader:
            print(f"  Roman shape: {roman_batch.shape}, Devanagari shape: {dev_batch.shape}")
            break
            
        print("\nTest batch:")
        for roman_batch, dev_batch in test_loader:
            print(f"  Roman shape: {roman_batch.shape}, Devanagari shape: {dev_batch.shape}")
            break
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_preprocessor()