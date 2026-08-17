import random
from collections import defaultdict

class HybridSampler:
    def __init__(self, seed=42, failure_analysis=None):
        self.ortho_to_examples = defaultdict(set)
        self.example_ortho_units = []
        self.seed = seed
        self.random_state = random.Random(seed)
        
        # Failure analysis integration
        self.failure_analysis = failure_analysis
        self.problematic_chars = set()
        self.problematic_units = set()
        self._initialize_problematic_elements()
    
    def _initialize_problematic_elements(self):
        """Initialize problematic elements from failure analysis"""
        if self.failure_analysis:
            # High-error characters (error rate > 0.6)
            for char, error_rate in self.failure_analysis.get('top_char_failures', []):
                if error_rate > 0.6:
                    self.problematic_chars.add(char)
            
            # High-error units (error rate > 0.8)
            for unit, error_rate in self.failure_analysis.get('top_unit_failures', []):
                if error_rate > 0.8:
                    self.problematic_units.add(unit)
            
            print(f"🎯 Loaded {len(self.problematic_chars)} problematic chars and {len(self.problematic_units)} problematic units")
    
    def _extract_orthographic_units(self, devanagari_word):
        """Same deterministic function as before"""
        characters = []
        i = 0
        n = len(devanagari_word)
        
        while i < n:
            current_char = devanagari_word[i]
            j = i + 1
            
            while j < n:
                next_char = devanagari_word[j]
                is_modifier = (
                    ('\u093E' <= next_char <= '\u094C') or
                    next_char == '\u094D' or next_char == '\u093C' or
                    next_char == '\u0902' or next_char == '\u0901' or next_char == '\u0903'
                )
                if is_modifier:
                    current_char += next_char
                    j += 1
                else:
                    break
            
            characters.append(current_char)
            i = j
        
        return characters
    
    def analyze_for_sampling(self, data_pairs):
        """Enhanced analysis that also tracks problematic elements"""
        self.ortho_to_examples.clear()
        self.example_ortho_units = []
        
        for i, (roman, devanagari) in enumerate(data_pairs):
            ortho_units = self._extract_orthographic_units(devanagari)
            self.example_ortho_units.append((i, set(ortho_units)))
            
            for unit in ortho_units:
                self.ortho_to_examples[unit].add(i)
    
    def failure_aware_sampling(self, data_pairs, target_size):
        """Sample focusing on examples with problematic characters/units"""
        print("🎯 Using failure-aware sampling...")
        
        scored_examples = []
        
        for i, (roman, devanagari) in enumerate(data_pairs):
            score = 0.0
            units = self._extract_orthographic_units(devanagari)
            
            # Score based on problematic elements
            for char in self.problematic_chars:
                if char in devanagari:
                    score += 3.0  # High weight for problematic chars
            
            for unit in self.problematic_units:
                if unit in units:
                    score += 4.0  # Even higher weight for problematic units
            
            # Bonus for multiple problematic elements
            problematic_count = sum(1 for char in self.problematic_chars if char in devanagari)
            score += problematic_count * 0.5
            
            if score > 0:
                scored_examples.append((score, i, roman, devanagari))
        
        # Sort by score and take top examples
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        selected_indices = [idx for _, idx, _, _ in scored_examples[:target_size]]
        
        # Calculate coverage of problematic elements
        covered_chars, covered_units = self._analyze_problematic_coverage([data_pairs[i] for i in selected_indices])
        
        print(f"✅ Failure-aware sampling: {len(selected_indices)} examples")
        print(f"   • Covered {len(covered_chars)}/{len(self.problematic_chars)} problematic chars")
        print(f"   • Covered {len(covered_units)}/{len(self.problematic_units)} problematic units")
        
        return [data_pairs[i] for i in selected_indices]
    
    def _analyze_problematic_coverage(self, sampled_data):
        """Analyze coverage of problematic elements in sampled data"""
        covered_chars = set()
        covered_units = set()
        
        for _, devanagari in sampled_data:
            # Check character coverage
            for char in self.problematic_chars:
                if char in devanagari:
                    covered_chars.add(char)
            
            # Check unit coverage
            units = self._extract_orthographic_units(devanagari)
            for unit in units:
                if unit in self.problematic_units:
                    covered_units.add(unit)
        
        return covered_chars, covered_units
    
    def enhanced_frequency_sampling(self, data_pairs, target_size):
        """Enhanced frequency sampling that weights problematic elements higher"""
        print("Using enhanced frequency-based sampling...")
        
        # Analyze units and their frequencies
        unit_frequencies = defaultdict(int)
        char_frequencies = defaultdict(int)
        
        for _, devanagari in data_pairs:
            units = self._extract_orthographic_units(devanagari)
            for unit in set(units):
                unit_frequencies[unit] += 1
            for char in set(devanagari):
                char_frequencies[char] += 1
        
        # Score each example with enhanced weights for problematic elements
        example_scores = []
        for i, (roman, devanagari) in enumerate(data_pairs):
            units = set(self._extract_orthographic_units(devanagari))
            chars = set(devanagari)
            
            # Base score: inverse frequency (prioritize rare units)
            base_score = sum(1.0 / (unit_frequencies[unit] + 1) for unit in units)
            base_score += sum(1.0 / (char_frequencies[char] + 1) for char in chars)
            
            # Enhancement: extra weight for problematic elements
            problem_score = 0.0
            for char in self.problematic_chars:
                if char in chars:
                    problem_score += 5.0 / (char_frequencies[char] + 1)  # 5x weight
            
            for unit in self.problematic_units:
                if unit in units:
                    problem_score += 7.0 / (unit_frequencies[unit] + 1)  # 7x weight
            
            total_score = base_score + problem_score
            example_scores.append((total_score, i))
        
        # Sort by score and take top examples
        example_scores.sort(reverse=True)
        selected_indices = [idx for _, idx in example_scores[:target_size]]
        
        # Calculate coverage for reporting
        covered_units = set()
        covered_chars = set()
        for idx in selected_indices:
            _, devanagari = data_pairs[idx]
            units = self._extract_orthographic_units(devanagari)
            covered_units.update(units)
            covered_chars.update(devanagari)
        
        # Analyze problematic coverage
        problem_chars_covered = len(self.problematic_chars & covered_chars)
        problem_units_covered = len(self.problematic_units & covered_units)
        
        print(f"✅ Enhanced frequency sampling: {len(selected_indices)} examples")
        print(f"   • Covers {problem_chars_covered}/{len(self.problematic_chars)} problematic chars")
        print(f"   • Covers {problem_units_covered}/{len(self.problematic_units)} problematic units")
        print(f"   • Total units covered: {len(covered_units)}/{len(unit_frequencies)}")
        
        return [data_pairs[i] for i in selected_indices], covered_units
    
    def stratified_sampling(self, data_pairs, target_size):
        """Deterministic stratified sampling"""
        buckets = {
            'short': (1, 4), 'medium': (5, 8), 'long': (9, 12), 'xlong': (13, 50)
        }
        
        # Group examples by length (deterministic)
        bucket_contents = {name: [] for name in buckets}
        for example in data_pairs:
            roman_word, _ = example
            length = len(roman_word)
            for bucket_name, (min_len, max_len) in buckets.items():
                if min_len <= length <= max_len:
                    bucket_contents[bucket_name].append(example)
                    break
        
        # Sort each bucket for deterministic sampling
        for bucket_name in bucket_contents:
            bucket_contents[bucket_name].sort()
        
        # Sample deterministically
        final_samples = []
        samples_per_bucket = target_size // len(buckets)
        
        for bucket_name, examples in bucket_contents.items():
            if len(examples) <= samples_per_bucket:
                final_samples.extend(examples)
            else:
                final_samples.extend(examples[:samples_per_bucket])
        
        return final_samples
    
    def greedy_set_cover(self, data_pairs, max_examples):
        """Greedy set cover focusing on problematic units first"""
        self.analyze_for_sampling(data_pairs)
        
        selected_indices = []
        covered_units = set()
        
        # First pass: prioritize problematic units
        remaining_problematic = self.problematic_units.copy()
        remaining_units = set(self.ortho_to_examples.keys())
        
        print(f"Total unique orthographic units: {len(remaining_units)}")
        print(f"Targeting {len(remaining_problematic)} problematic units")
        
        # Phase 1: Cover problematic units
        while remaining_problematic and len(selected_indices) < max_examples // 2:
            best_example = None
            best_problematic_coverage = 0
            
            for example_idx, units in self.example_ortho_units:
                if example_idx in selected_indices:
                    continue
                    
                uncovered_problematic = units & remaining_problematic
                if len(uncovered_problematic) > best_problematic_coverage:
                    best_problematic_coverage = len(uncovered_problematic)
                    best_example = example_idx
            
            if best_example is None:
                break
            
            selected_indices.append(best_example)
            covered_units.update(self.example_ortho_units[best_example][1])
            remaining_problematic = remaining_problematic - covered_units
            remaining_units = remaining_units - covered_units
        
        # Phase 2: Cover remaining units
        while remaining_units and len(selected_indices) < max_examples:
            best_example = None
            best_coverage = 0
            
            for example_idx, units in self.example_ortho_units:
                if example_idx in selected_indices:
                    continue
                    
                uncovered = units - covered_units
                if len(uncovered) > best_coverage:
                    best_coverage = len(uncovered)
                    best_example = example_idx
            
            if best_example is None:
                break
            
            selected_indices.append(best_example)
            covered_units.update(self.example_ortho_units[best_example][1])
            remaining_units = remaining_units - covered_units
        
        final_samples = [data_pairs[i] for i in selected_indices]
        return final_samples, covered_units, selected_indices
    
    def hybrid_sampling(self, data_pairs, target_size=100000, method="enhanced"):
        """Enhanced hybrid sampling with failure awareness"""
        print(f"Starting enhanced hybrid sampling (method: {method})...")
        
        if method == "enhanced":
            # Use enhanced frequency-based sampling
            print("Phase 1: Enhanced frequency-based sampling...")
            greedy_samples, covered_units = self.enhanced_frequency_sampling(data_pairs, target_size // 2)
        elif method == "failure_aware":
            # Use failure-aware sampling
            print("Phase 1: Failure-aware sampling...")
            greedy_samples = self.failure_aware_sampling(data_pairs, target_size // 2)
            covered_units = set()
            for _, devanagari in greedy_samples:
                units = self._extract_orthographic_units(devanagari)
                covered_units.update(units)
        else:
            # Fall back to original frequency sampling
            print("Phase 1: Standard frequency-based sampling...")
            greedy_samples, covered_units = self.frequency_based_sampling(data_pairs, target_size // 2)
        
        print(f"Covered {len(covered_units)} units with {len(greedy_samples)} examples")
        
        # Get remaining data
        greedy_set = set(greedy_samples)
        remaining_data = [ex for ex in data_pairs if ex not in greedy_set]
        
        print(f"Phase 2: Stratified sampling...")
        stratified_samples = self.stratified_sampling(remaining_data, target_size // 3)
        
        # Get final remaining data
        stratified_set = set(stratified_samples)
        final_remaining = [ex for ex in remaining_data if ex not in stratified_set]
        
        print(f"Phase 3: Random sampling...")
        random_needed = target_size - len(greedy_samples) - len(stratified_samples)
        random_samples = self.random_state.sample(final_remaining, min(len(final_remaining), random_needed))
        
        final_samples = greedy_samples + stratified_samples + random_samples
        
        # Final coverage analysis
        self._analyze_final_coverage(final_samples)
        
        print(f"Final dataset: {len(final_samples)} examples")
        return final_samples
    
    def _analyze_final_coverage(self, final_samples):
        """Analyze final coverage of problematic elements"""
        if not self.problematic_chars and not self.problematic_units:
            return
        
        covered_chars = set()
        covered_units = set()
        
        for _, devanagari in final_samples:
            covered_chars.update(devanagari)
            units = self._extract_orthographic_units(devanagari)
            covered_units.update(units)
        
        problem_chars_covered = len(self.problematic_chars & covered_chars)
        problem_units_covered = len(self.problematic_units & covered_units)
        
        print(f"📊 FINAL COVERAGE ANALYSIS:")
        print(f"   • Problematic chars: {problem_chars_covered}/{len(self.problematic_chars)} ({problem_chars_covered/len(self.problematic_chars):.1%})")
        print(f"   • Problematic units: {problem_units_covered}/{len(self.problematic_units)} ({problem_units_covered/len(self.problematic_units):.1%})")
        
        # Show top missing problematic elements
        missing_chars = self.problematic_chars - covered_chars
        missing_units = self.problematic_units - covered_units
        
        if missing_chars:
            print(f"   • Top missing chars: {list(missing_chars)[:5]}")
        if missing_units:
            print(f"   • Top missing units: {list(missing_units)[:5]}")
    
    # Keep the original frequency_based_sampling for fallback
    def frequency_based_sampling(self, data_pairs, target_size):
        """Original frequency-based sampling"""
        print("Using frequency-based sampling (fast)...")
        
        unit_frequencies = defaultdict(int)
        for _, devanagari in data_pairs:
            units = self._extract_orthographic_units(devanagari)
            for unit in set(units):
                unit_frequencies[unit] += 1
        
        example_scores = []
        for i, (roman, devanagari) in enumerate(data_pairs):
            units = set(self._extract_orthographic_units(devanagari))
            score = sum(1.0 / unit_frequencies[unit] for unit in units)
            example_scores.append((score, i))
        
        example_scores.sort(reverse=True)
        selected_indices = [idx for _, idx in example_scores[:target_size]]
        
        covered_units = set()
        for idx in selected_indices:
            _, devanagari = data_pairs[idx]
            units = self._extract_orthographic_units(devanagari)
            covered_units.update(units)
        
        print(f"✅ Frequency sampling: {len(selected_indices)} examples cover {len(covered_units)} units")
        
        return [data_pairs[i] for i in selected_indices], covered_units

    def save_sampled_indices(self, data_pairs, output_path):
        """Save the indices of sampled examples for exact reproducibility"""
        sampled_data = self.hybrid_sampling(data_pairs)
        
        sampled_indices = []
        for sample in sampled_data:
            try:
                index = data_pairs.index(sample)
                sampled_indices.append(index)
            except ValueError:
                pass
        
        with open(output_path, 'w') as f:
            for idx in sampled_indices:
                f.write(f"{idx}\n")
        
        print(f"Saved {len(sampled_indices)} indices to {output_path}")
        return sampled_indices

    def load_sampled_indices(self, data_pairs, indices_path):
        """Load previously sampled indices"""
        with open(indices_path, 'r') as f:
            indices = [int(line.strip()) for line in f]
        
        sampled_data = [data_pairs[i] for i in indices]
        print(f"Loaded {len(sampled_data)} examples from {indices_path}")
        return sampled_data


# Updated usage example with failure analysis
def demonstrate_enhanced_sampler():
    """Demonstrate the enhanced sampler with failure analysis"""
    
    # Your actual failure analysis results
    failure_analysis = {
        'top_char_failures': [
            ["ॅ", 1.0], ["ऋ", 1.0], ["ऑ", 0.95], ["ऐ", 0.81], ["ऊ", 0.81],
            ["ँ", 0.77], ["इ", 0.73], ["फ", 0.72], ["ड", 0.72], ["ॉ", 0.72]
        ],
        'top_unit_failures': [
            ["फ़ि", 1.0], ["यॉ", 1.0], ["डं", 1.0], ["ड़ि", 1.0], ["सां", 1.0],
            ["योंं", 1.0], ["फाॅ", 1.0], ["साे", 1.0], ["काॅ", 1.0], ["शाॅ", 1.0]
        ]
    }
    
    # Create test data that includes some problematic elements
    test_data = [
        # Examples with problematic characters
        ("film", "फ़िल्म"),  # Contains फ़
        ("yog", "योग"),      # Normal case
        ("danda", "डंडा"),   # Contains डं
        ("sanskar", "संस्कार"), # Normal case
        ("jhoom", "झूम"),    # Contains झ
        ("tra", "त्र"),      # Contains त्र
        ("shri", "श्री"),    # Normal case
        ("om", "ओम"),       # Normal case
        ("ain", "ऐन"),      # Contains ऐ (problematic)
        ("ook", "ऊक"),      # Contains ऊ (problematic)
    ]
    
    print("🧪 DEMONSTRATING ENHANCED SAMPLER")
    print("=" * 50)
    
    # Create sampler with failure analysis
    sampler = HybridSampler(failure_analysis=failure_analysis)
    
    # Test different sampling methods
    print("\n1. Testing enhanced frequency sampling:")
    enhanced_samples, _ = sampler.enhanced_frequency_sampling(test_data, 5)
    
    print("\n2. Testing failure-aware sampling:")
    failure_aware_samples = sampler.failure_aware_sampling(test_data, 5)
    
    print("\n3. Testing enhanced hybrid sampling:")
    hybrid_samples = sampler.hybrid_sampling(test_data, 8, method="enhanced")
    
    print(f"\n📊 Demonstration completed!")
    print(f"   • Enhanced frequency samples: {len(enhanced_samples)}")
    print(f"   • Failure-aware samples: {len(failure_aware_samples)}")
    print(f"   • Hybrid samples: {len(hybrid_samples)}")


# if __name__ == "__main__":
#     # Run the original tests
#     from hybrid_sampler import test_orthographic_splitting, better_test_sampler, stress_test_sampler
    
#     print("Testing Enhanced Hybrid Sampler with Failure Analysis")
#     print("=" * 60)
    
#     # Run original tests
#     test_orthographic_splitting()
#     print()
#     better_test_sampler()
#     stress_test_sampler()
    
#     # Run enhanced demonstration
#     print("\n" + "=" * 60)
#     demonstrate_enhanced_sampler()