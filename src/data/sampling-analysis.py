import matplotlib.pyplot as plt
from collections import Counter
from sampler import HybridSampler
import json
import os

# plt.rcParams['font.family'] = 'Noto Sans Devanagari'  # or 'Mangal', 'Lohit Devanagari'
# plt.rcParams['font.size'] = 10


def load_jsonl_data():
        """Load data from JSONL file (Aksharantar format)"""
        data_pairs = []
        
        # Construct full path
        # data_dir = 'data/raw/aksharantar'
        full_path = '../../data/raw/hin/hin_train.json'
        
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
        
        print(f"✅ Loaded {len(data_pairs)} valid examples from {full_path}")
        return data_pairs



def plot_character_distribution(full_data, sampled_data, sampler, 
                                top_k=30, mode='unit', 
                                save_path=None, show_plot=False):
    """
    Plot normalized distribution of characters or orthographic units
    in full vs sampled data, with x-axis in Devanagari and labels in English.
    """

    # Collect units or characters
    full_units, sampled_units = [], []
    for _, dev in full_data:
        if mode == 'unit':
            full_units.extend(sampler._extract_orthographic_units(dev))
        else:
            full_units.extend(list(dev))
    for _, dev in sampled_data:
        if mode == 'unit':
            sampled_units.extend(sampler._extract_orthographic_units(dev))
        else:
            sampled_units.extend(list(dev))

    # Count and normalize
    full_counter = Counter(full_units)
    sampled_counter = Counter(sampled_units)
    full_total = sum(full_counter.values())
    sampled_total = sum(sampled_counter.values())
    full_freq = {ch: c / full_total for ch, c in full_counter.items()}
    sampled_freq = {ch: sampled_counter.get(ch, 0) / sampled_total for ch in full_counter.keys()}

    # Select top-K
    top_units = [u for u, _ in Counter(full_counter).most_common(top_k)]
    full_y = [full_freq[u] for u in top_units]
    sampled_y = [sampled_freq.get(u, 0) for u in top_units]

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_units)), full_y, width=0.4, label="Full Data", align='center')
    plt.bar([i + 0.4 for i in range(len(top_units))], sampled_y, width=0.4, label="Sampled Data", align='center')

    # X-axis labels in Devanagari
    plt.xticks([i + 0.2 for i in range(len(top_units))], top_units, 
               rotation=90, fontname='Noto Sans Devanagari')

    # Titles and labels in English
    plt.xlabel("Devanagari Character" if mode == 'char' else "Orthographic Unit")
    plt.ylabel("Normalized Frequency")
    plt.title(f"Top {top_k} Most Frequent {'Characters' if mode == 'char' else 'Orthographic Units'}")

    # Legend in English
    plt.legend()

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()





train_pairs = load_jsonl_data()
sample_size = 100000
sampler = HybridSampler()
sampled_data = sampler.hybrid_sampling(train_pairs, sample_size, method='fast')


plot_character_distribution(
    full_data=train_pairs,
    sampled_data=sampled_data,
    sampler=sampler,
    top_k=100,
    mode='unit',
    save_path='plots/unit_hybrid_distribution.png'
)

plot_character_distribution(
    full_data=train_pairs,
    sampled_data=sampled_data,
    sampler=sampler,
    top_k=100,
    mode='char',
    save_path='plots/char_hybrid_distribution.png'
)