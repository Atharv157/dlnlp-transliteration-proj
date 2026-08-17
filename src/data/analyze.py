import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import sys

# Add your source directory to path to import HybridSampler
sys.path.append('src')

from src.data.sampler import HybridSampler

class SamplingAnalyzer:
    def __init__(self, sampler):
        self.sampler = sampler
        self.results = {}
    
    def load_jsonl_data(self, file_path):
        """Load data from JSONL file (Aksharantar format)"""
        data_pairs = []
        
        print(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
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

    def analyze_ortho_units_distribution(self, full_data, subsampled_data):
        """Comprehensive analysis of orthographic unit distributions"""
        
        print("🔍 Analyzing orthographic unit distributions...")
        
        # Extract ortho units from both datasets
        full_units = self._extract_all_units(full_data)
        subsampled_units = self._extract_all_units(subsampled_data)
        
        # 1. Coverage Analysis
        coverage_metrics = self._calculate_coverage(full_units, subsampled_units)
        
        # 2. Frequency Distribution Comparison  
        freq_metrics = self._compare_frequency_distributions(full_units, subsampled_units)
        
        # 3. Rare Units Analysis
        rare_units_metrics = self._analyze_rare_units(full_units, subsampled_units)
        
        # 4. Efficiency Metrics
        efficiency_metrics = self._calculate_efficiency(full_data, subsampled_data, coverage_metrics)
        
        # Combine all results
        analysis_results = {
            'coverage': coverage_metrics,
            'frequency_distribution': freq_metrics,
            'rare_units': rare_units_metrics,
            'efficiency': efficiency_metrics,
            'full_units_stats': full_units,
            'subsampled_units_stats': subsampled_units
        }
        
        self.results = analysis_results
        return analysis_results
    
    def _extract_all_units(self, data_pairs):
        """Extract ortho units from all data pairs with frequencies"""
        unit_counter = Counter()
        all_units = set()
        
        for roman, devanagari in data_pairs:
            units = self.sampler._extract_orthographic_units(devanagari)
            unit_counter.update(units)
            all_units.update(units)
        
        return {
            'counter': unit_counter,
            'unique_units': all_units,
            'total_occurrences': sum(unit_counter.values()),
            'most_common': unit_counter.most_common(20)  # Top 20 units
        }
    
    def _calculate_coverage(self, full_units, subsampled_units):
        """Calculate coverage metrics"""
        full_unique = full_units['unique_units']
        sub_unique = subsampled_units['unique_units']
        
        covered_units = full_unique.intersection(sub_unique)
        missing_units = full_unique - sub_unique
        
        return {
            'full_unique_count': len(full_unique),
            'subsampled_unique_count': len(sub_unique),
            'covered_count': len(covered_units),
            'coverage_ratio': len(covered_units) / len(full_unique),
            'missing_count': len(missing_units),
            'missing_units': list(missing_units)[:20]  # First 20 missing units
        }
    
    def _compare_frequency_distributions(self, full_units, subsampled_units):
        """Compare frequency distributions between full and subsampled data"""
        full_counter = full_units['counter']
        sub_counter = subsampled_units['counter']
        
        # Get common units and their frequencies
        common_units = set(full_counter.keys()).intersection(set(sub_counter.keys()))
        
        full_freqs = []
        sub_freqs = []
        
        for unit in common_units:
            full_freqs.append(full_counter[unit])
            sub_freqs.append(sub_counter[unit])
        
        # Calculate correlation
        if len(full_freqs) > 1:
            correlation, p_value = pearsonr(full_freqs, sub_freqs)
        else:
            correlation, p_value = 0, 1
        
        return {
            'common_units_count': len(common_units),
            'frequency_correlation': correlation,
            'correlation_p_value': p_value,
            'full_total_occurrences': full_units['total_occurrences'],
            'sub_total_occurrences': subsampled_units['total_occurrences']
        }
    
    def _analyze_rare_units(self, full_units, subsampled_units, rare_threshold=5):
        """Analyze preservation of rare ortho units"""
        full_counter = full_units['counter']
        
        # Identify rare units (appearing <= rare_threshold times)
        rare_units = {unit for unit, count in full_counter.items() if count <= rare_threshold}
        preserved_rare_units = rare_units.intersection(subsampled_units['unique_units'])
        
        return {
            'rare_units_total': len(rare_units),
            'rare_units_preserved': len(preserved_rare_units),
            'rare_units_preservation_ratio': len(preserved_rare_units) / len(rare_units) if rare_units else 0,
            'rare_threshold': rare_threshold
        }
    
    def _calculate_efficiency(self, full_data, subsampled_data, coverage_metrics):
        """Calculate sampling efficiency metrics"""
        full_size = len(full_data)
        sub_size = len(subsampled_data)
        
        return {
            'full_dataset_size': full_size,
            'subsampled_size': sub_size,
            'compression_ratio': sub_size / full_size,
            'coverage_efficiency': coverage_metrics['covered_count'] / sub_size,
            'units_per_example': coverage_metrics['covered_count'] / sub_size
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            print("No analysis results available. Run analyze_ortho_units_distribution first.")
            return
        
        print("\n" + "="*80)
        print("📊 HYBRID SAMPLING ANALYSIS REPORT")
        print("="*80)
        
        cov = self.results['coverage']
        freq = self.results['frequency_distribution']
        rare = self.results['rare_units']
        eff = self.results['efficiency']
        
        print(f"\n📈 COVERAGE ANALYSIS:")
        print(f"   • Full dataset unique units: {cov['full_unique_count']}")
        print(f"   • Subsampled unique units: {cov['subsampled_unique_count']}")
        print(f"   • Units covered: {cov['covered_count']} ({cov['coverage_ratio']:.1%})")
        print(f"   • Units missing: {cov['missing_count']}")
        
        print(f"\n📊 FREQUENCY DISTRIBUTION:")
        print(f"   • Common units analyzed: {freq['common_units_count']}")
        print(f"   • Frequency correlation: {freq['frequency_correlation']:.3f}")
        print(f"   • Correlation significance: p = {freq['correlation_p_value']:.4f}")
        
        print(f"\n🎯 RARE UNITS PRESERVATION (≤{rare['rare_threshold']} occurrences):")
        print(f"   • Total rare units: {rare['rare_units_total']}")
        print(f"   • Rare units preserved: {rare['rare_units_preserved']}")
        print(f"   • Preservation ratio: {rare['rare_units_preservation_ratio']:.1%}")
        
        print(f"\n⚡ EFFICIENCY METRICS:")
        print(f"   • Full dataset size: {eff['full_dataset_size']}")
        print(f"   • Subsampled size: {eff['subsampled_size']}")
        print(f"   • Compression ratio: {eff['compression_ratio']:.1%}")
        print(f"   • Coverage efficiency: {eff['coverage_efficiency']:.2f} units/example")
        print(f"   • Units per example: {eff['units_per_example']:.2f}")
        
        # Show some examples
        print(f"\n🔤 SAMPLE ORTHO UNITS (Top 10 most common in full dataset):")
        for unit, count in self.results['full_units_stats']['most_common'][:10]:
            print(f"   • '{unit}': {count} occurrences")
        
        if cov['missing_units']:
            print(f"\n⚠️  EXAMPLE MISSING UNITS (first 10):")
            for unit in cov['missing_units'][:10]:
                print(f"   • '{unit}'")

    def plot_distribution_comparison(self, save_path=None):
        """Plot comparison of unit frequency distributions"""
        if not self.results:
            print("No results to plot")
            return
        
        full_counter = self.results['full_units_stats']['counter']
        sub_counter = self.results['subsampled_units_stats']['counter']
        
        # Get common units
        common_units = set(full_counter.keys()).intersection(set(sub_counter.keys()))
        
        # Prepare data for plotting
        units = list(common_units)
        full_freqs = [full_counter[unit] for unit in units]
        sub_freqs = [sub_counter[unit] for unit in units]
        
        # Take top 50 for readability
        if len(units) > 50:
            # Sort by full dataset frequency and take top 50
            combined = list(zip(units, full_freqs, sub_freqs))
            combined.sort(key=lambda x: x[1], reverse=True)
            units, full_freqs, sub_freqs = zip(*combined[:50])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Side-by-side comparison
        x = np.arange(len(units))
        width = 0.35
        
        ax1.bar(x - width/2, full_freqs, width, label='Full Dataset', alpha=0.7)
        ax1.bar(x + width/2, sub_freqs, width, label='Subsampled', alpha=0.7)
        ax1.set_xlabel('Orthographic Units')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Frequency Distribution: Full Dataset vs Subsampled')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(units, rotation=45, ha='right')
        
        # Plot 2: Scatter plot correlation
        ax2.scatter(full_freqs, sub_freqs, alpha=0.6)
        ax2.plot([0, max(full_freqs)], [0, max(full_freqs)], 'r--', alpha=0.8)
        ax2.set_xlabel('Full Dataset Frequency')
        ax2.set_ylabel('Subsampled Frequency')
        ax2.set_title(f'Frequency Correlation (r = {self.results["frequency_distribution"]["frequency_correlation"]:.3f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()

def main():
    """Main analysis script"""
    # Configuration
    TRAIN_FILE_PATH = "data/raw/hin/hin_train.json"  # Update this path
    SAMPLE_SIZE = 100000  # Adjust based on your needs
    SAMPLING_METHOD = "fast"
    
    print("🚀 Starting Hybrid Sampling Analysis")
    print("="*50)
    
    # Initialize sampler and analyzer
    sampler = HybridSampler(seed=42)
    analyzer = SamplingAnalyzer(sampler)
    
    # Load full training data
    print("📥 Loading training data...")
    full_train_pairs = analyzer.load_jsonl_data(TRAIN_FILE_PATH)
    
    if not full_train_pairs:
        print("❌ Failed to load training data")
        return
    
    print(f"📊 Full dataset size: {len(full_train_pairs)} examples")
    
    # Create subsample using your hybrid sampling
    print("\n🎯 Running hybrid sampling...")
    subsampled_pairs = sampler.hybrid_sampling(
        full_train_pairs, 
        SAMPLE_SIZE, 
        method=SAMPLING_METHOD
    )
    
    print(f"✅ Subsampled dataset size: {len(subsampled_pairs)} examples")
    
    # Analyze the sampling quality
    print("\n🔍 Analyzing sampling quality...")
    analysis_results = analyzer.analyze_ortho_units_distribution(
        full_train_pairs, 
        subsampled_pairs
    )
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Create visualization
    print("\n📈 Generating visualization...")
    analyzer.plot_distribution_comparison("sampling_analysis_plot.png")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()