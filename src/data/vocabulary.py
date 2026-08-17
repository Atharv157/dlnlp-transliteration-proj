class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.n_chars = 4
    
    def add_text(self, text):
        """Add characters from text to vocabulary"""
        for char in text:
            if char not in self.char2idx:
                self.char2idx[char] = self.n_chars
                self.idx2char[self.n_chars] = char
                self.n_chars += 1
    
    def encode(self, text):
        """Convert text to indices, adding SOS and EOS tokens"""
        indices = [self.char2idx.get(char, self.char2idx['<unk>']) for char in text]
        return [self.char2idx['<sos>']] + indices + [self.char2idx['<eos>']]
    
    def decode(self, indices):
        """Convert indices back to text"""
        chars = []
        for idx in indices:
            if idx in [self.char2idx['<sos>'], self.char2idx['<eos>'], self.char2idx['<pad>']]:
                continue
            chars.append(self.idx2char.get(idx, '<unk>'))
        return ''.join(chars)
    
    def __len__(self):
        return self.n_chars

class RomanVocabulary(Vocabulary):
    def __init__(self):
        super().__init__("roman")
    
    @staticmethod
    def split_roman_word(word):
        """Split Roman word into characters (lowercase)"""
        return list(word.lower())

class DevanagariVocabulary(Vocabulary):
    def __init__(self):
        super().__init__("devanagari")
    
    # @staticmethod
    # def split_devanagari_word(devanagari_word):
    #     """Same deterministic function as before"""
    #     characters = []
    #     i = 0
    #     n = len(devanagari_word)
        
    #     while i < n:
    #         current_char = devanagari_word[i]
    #         j = i + 1
            
    #         while j < n:
    #             next_char = devanagari_word[j]
    #             is_modifier = (
    #                 ('\u093E' <= next_char <= '\u094C') or
    #                 next_char == '\u094D' or next_char == '\u093C' or
    #                 next_char == '\u0902' or next_char == '\u0901' or next_char == '\u0903'
    #             )
    #             if is_modifier:
    #                 current_char += next_char
    #                 j += 1
    #             else:
    #                 break
            
    #         characters.append(current_char)
    #         i = j
        
    #     return characters
    
    @staticmethod
    def split_devanagari_word(word):
        """Split Devanagari word into simple Unicode characters - SIMPLE APPROACH"""
        return list(word)

    # @staticmethod
    # def split_devanagari_word(word):
    #     """Split Devanagari word into orthographic syllables"""
    #     characters = []
    #     current_char = ""
        
    #     for char in word:
    #         # Check if character is a modifier (vowel sign, virama, nukta)
    #         is_modifier = ('\u093E' <= char <= '\u094C') or char == '\u094D' or char == '\u093C'
            
    #         if current_char and is_modifier:
    #             # Combine modifier with base character
    #             current_char += char
    #             characters.append(current_char)
    #             current_char = ""
    #         else:
    #             if current_char:
    #                 characters.append(current_char)
    #             current_char = char
        
    #     if current_char:
    #         characters.append(current_char)
        
    #     return characters

# Test the implementation
if __name__ == "__main__":
    # Test Devanagari splitting
    test_words = ["जन्मदिवस", "रक्खा", "मिलीजुली"]
    dev_vocab = DevanagariVocabulary()
    
    for word in test_words:
        syllables = dev_vocab.split_devanagari_word(word)
        print(f"{word} → {syllables}")
    
    # Test Roman splitting
    roman_vocab = RomanVocabulary()
    roman_words = ["janamdivas", "rakha", "milijuli"]
    
    for word in roman_words:
        chars = roman_vocab.split_roman_word(word)
        print(f"{word} → {chars}")