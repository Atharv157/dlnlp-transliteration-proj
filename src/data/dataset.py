import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TransliterationDataset(Dataset):
    def __init__(self, data_pairs, roman_vocab, devanagari_vocab):
        self.data_pairs = data_pairs
        self.roman_vocab = roman_vocab
        self.devanagari_vocab = devanagari_vocab
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        roman_word, devanagari_word = self.data_pairs[idx]
        
        # Split into characters/syllables
        roman_chars = self.roman_vocab.split_roman_word(roman_word)
        devanagari_chars = self.devanagari_vocab.split_devanagari_word(devanagari_word)
        
        # Encode to indices
        roman_encoded = self.roman_vocab.encode(roman_chars)
        devanagari_encoded = self.devanagari_vocab.encode(devanagari_chars)
        
        return (
            torch.tensor(roman_encoded, dtype=torch.long),
            torch.tensor(devanagari_encoded, dtype=torch.long)
        )

def collate_fn(batch):
    """Custom collate function to pad variable length sequences and return lengths"""
    roman_sequences, devanagari_sequences = zip(*batch)
    
    # Get sequence lengths before padding
    roman_lengths = [len(seq) for seq in roman_sequences]
    devanagari_lengths = [len(seq) for seq in devanagari_sequences]
    
    # Pad sequences
    roman_padded = pad_sequence(roman_sequences, batch_first=True, padding_value=0)
    devanagari_padded = pad_sequence(devanagari_sequences, batch_first=True, padding_value=0)
    
    return (
        roman_padded, 
        devanagari_padded,
        torch.tensor(roman_lengths, dtype=torch.long),
        torch.tensor(devanagari_lengths, dtype=torch.long)
    )


# def collate_fn(batch):
#     """Custom collate function to pad variable length sequences"""
#     roman_sequences, devanagari_sequences = zip(*batch)
    
#     # Pad sequences
#     roman_padded = pad_sequence(roman_sequences, batch_first=True, padding_value=0)
#     devanagari_padded = pad_sequence(devanagari_sequences, batch_first=True, padding_value=0)
    
#     return roman_padded, devanagari_padded

# Test the dataset
def test_dataset():
    from vocabulary import RomanVocabulary, DevanagariVocabulary
    
    # Test data
    test_pairs = [
        ("janamdivas", "जन्मदिवस"),
        ("rakha", "रक्खा"),
        ("milijuli", "मिलीजुली")
    ]
    
    # Build vocabularies
    roman_vocab = RomanVocabulary()
    devanagari_vocab = DevanagariVocabulary()
    
    for roman, devanagari in test_pairs:
        roman_chars = roman_vocab.split_roman_word(roman)
        devanagari_chars = devanagari_vocab.split_devanagari_word(devanagari)
        roman_vocab.add_text(roman_chars)
        devanagari_vocab.add_text(devanagari_chars)
    
    # Create dataset
    dataset = TransliterationDataset(test_pairs, roman_vocab, devanagari_vocab)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test one item
    roman_seq, dev_seq = dataset[0]
    print(f"Roman sequence: {roman_seq}")
    print(f"Devanagari sequence: {dev_seq}")
    
    # Test collate function
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    for batch_idx, (roman_batch, devanagari_batch) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Roman shape: {roman_batch.shape}")
        print(f"  Devanagari shape: {devanagari_batch.shape}")
        break

if __name__ == "__main__":
    test_dataset()