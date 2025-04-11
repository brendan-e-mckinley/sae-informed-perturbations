import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os
from pathlib import Path
import requests
import json
from sae_lens import SAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WordConceptAnalyzer:
    def __init__(self, model_name="gpt2", layer=7):
        """
        Initialize the word concept analyzer with SAELens.
        
        Args:
            model_name: Name of the model (default: "gpt2")
            layer: Layer to analyze (default: 7)
        """
        self.model_name = model_name
        self.layer = layer
        
        # Load transformers model and tokenizer
        print(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        
        # Load SAE model from SAELens
        print(f"Loading SAELens model for {model_name}, layer {layer}...")
        self.sae_model = SAE.from_pretrained(
            release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
            sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
            device = "cuda"
        )
        
        print(f"SAELens model loaded with {len(self.sae_model.features)} features")
        
        # Hook for capturing activations
        self.activations = None
        self.hook_handle = None
        
        # Set up hook
        self._set_activation_hook()
    
    def _set_activation_hook(self):
        """Set up a hook to capture activations at the specified layer"""
        def hook_fn(module, input, output):
            self.activations = output.detach()
        
        # Access the specific layer
        transformer_block = self.model.transformer.h[self.layer]
        # Attach hook to the output of the MLP
        self.hook_handle = transformer_block.mlp.c_proj.register_forward_hook(hook_fn)
    
    def analyze_word(self, word, context_templates=None, top_n=10):
        """
        Analyze a word across different contexts to find its concept vectors.
        
        Args:
            word: The word to analyze
            context_templates: List of templates where {word} will be replaced
            top_n: Number of top features to return
            
        Returns:
            Dictionary of analysis results
        """
        if context_templates is None:
            context_templates = [
                "The {word} is",
                "I saw a {word} in the",
                "The {word} was very",
                "A {word} can be described as",
                "When thinking about {word}, I"
            ]
        
        # Replace {word} with the actual word in each template
        contexts = [template.format(word=word) for template in context_templates]
        
        # Track activations across all contexts
        all_word_activations = []
        
        for context in contexts:
            # Encode text
            inputs = self.tokenizer(context, return_tensors="pt").to(device)
            
            # Run model (we don't need the output, just for the hook to capture activations)
            with torch.no_grad():
                self.model(**inputs)
            
            # Get word token position
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            
            if len(word_tokens) > 1:
                print(f"Warning: '{word}' is tokenized into multiple tokens: {self.tokenizer.convert_ids_to_tokens(word_tokens)}")
            
            # Find the position of word tokens in the context
            input_ids = inputs["input_ids"][0].tolist()
            
            # Find all occurrences of the first token of the word
            positions = []
            for i in range(len(input_ids) - len(word_tokens) + 1):
                if input_ids[i:i+len(word_tokens)] == word_tokens:
                    positions.extend(list(range(i, i+len(word_tokens))))
            
            if not positions:
                print(f"Warning: Could not find '{word}' tokens in context: '{context}'")
                continue
            
            # Get activations for the word tokens
            for pos in positions:
                if pos < self.activations.shape[1]:  # Ensure position is valid
                    token_activations = self.activations[0, pos, :].cpu().numpy()
                    all_word_activations.append(token_activations)
        
        # Combine feature activations across all contexts
        if not all_word_activations:
            return {"error": f"Could not analyze word '{word}' in any context"}
        
        # Average activations across all occurrences
        combined_activations = np.mean(all_word_activations, axis=0)
        
        # Use SAELens to get feature activations
        feature_activations = self.sae_model.encode_activations(combined_activations)
        
        # Get top features
        top_indices = np.argsort(-feature_activations)[:top_n]
        top_values = feature_activations[top_indices]
        
        # Get feature descriptions
        feature_descriptions = []
        for idx in top_indices:
            feature = self.sae_model.features[idx]
            desc = feature.name if hasattr(feature, 'name') and feature.name else f"Feature {idx}"
            feature_descriptions.append(desc)
        
        return {
            "word": word,
            "top_feature_indices": top_indices.tolist(),
            "top_feature_activations": top_values.tolist(),
            "top_feature_descriptions": feature_descriptions,
            "contexts_analyzed": contexts
        }
    
    def visualize_word_concepts(self, word, results=None, show_top_n=10):
        """
        Visualize the top concept features for a word.
        
        Args:
            word: The word that was analyzed
            results: Analysis results (if None, will run analyze_word)
            show_top_n: Number of top features to show
        """
        if results is None:
            results = self.analyze_word(word, top_n=show_top_n)
        
        if "error" in results:
            print(results["error"])
            return
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Get data to plot
        indices = results["top_feature_indices"][:show_top_n]
        values = results["top_feature_activations"][:show_top_n]
        descriptions = results["top_feature_descriptions"][:show_top_n]
        
        # Replace long descriptions with truncated ones
        short_descriptions = []
        for desc in descriptions:
            if len(desc) > 30:
                short_descriptions.append(desc[:27] + "...")
            else:
                short_descriptions.append(desc)
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(values)), values, align='center')
        plt.yticks(range(len(values)), [f"{idx}: {desc}" for idx, desc in zip(indices, short_descriptions)])
        plt.xlabel('Feature Activation')
        plt.title(f'Top Concept Features for "{word}"')
        
        # Add value labels to the right of bars
        for i, v in enumerate(values):
            plt.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_examples(self, feature_idx, top_n=5):
        """
        Get examples and information about a specific feature.
        
        Args:
            feature_idx: Index of the feature to examine
            top_n: Number of top examples to show
            
        Returns:
            Dictionary with feature information
        """
        try:
            feature = self.sae_model.features[feature_idx]
            
            # Get feature information from SAELens
            info = {
                "feature_index": feature_idx,
                "name": feature.name if hasattr(feature, 'name') else f"Feature {feature_idx}",
                "description": feature.description if hasattr(feature, 'description') else "No description available"
            }
            
            # Get examples if available
            if hasattr(feature, 'examples') and feature.examples:
                info["examples"] = feature.examples[:top_n]
            else:
                info["examples"] = ["No examples available"]
                
            return info
            
        except IndexError:
            return {"error": f"Feature index {feature_idx} is out of range"}
    
    def compare_words(self, words, top_n=10):
        """
        Compare concept features across multiple words.
        
        Args:
            words: List of words to compare
            top_n: Number of top features to consider
            
        Returns:
            Dictionary with comparison results
        """
        # Analyze each word
        word_results = {}
        all_features = set()
        
        for word in words:
            results = self.analyze_word(word, top_n=top_n)
            if "error" not in results:
                word_results[word] = results
                all_features.update(results["top_feature_indices"])
        
        # Create feature comparison data
        comparison = {
            "words": words,
            "feature_overlap": {},
            "unique_features": {},
            "shared_features": []
        }
        
        # Find shared features
        if len(words) > 1:
            # Identify features that appear in all words
            potential_shared = set(word_results[words[0]]["top_feature_indices"])
            for word in words[1:]:
                if word in word_results:
                    potential_shared &= set(word_results[word]["top_feature_indices"])
            
            shared_features = list(potential_shared)
            
            # Get descriptions for shared features
            shared_descriptions = []
            for feat_idx in shared_features:
                feature = self.sae_model.features[feat_idx]
                desc = feature.name if hasattr(feature, 'name') and feature.name else f"Feature {feat_idx}"
                shared_descriptions.append(desc)
            
            comparison["shared_features"] = [
                {"feature": feat, "description": desc}
                for feat, desc in zip(shared_features, shared_descriptions)
            ]
        
        # Calculate unique features for each word
        for word in words:
            if word not in word_results:
                continue
                
            # Get features unique to this word
            word_features = set(word_results[word]["top_feature_indices"])
            unique_features = word_features.copy()
            
            for other_word in words:
                if other_word != word and other_word in word_results:
                    other_features = set(word_results[other_word]["top_feature_indices"])
                    unique_features -= other_features
            
            # Get descriptions for unique features
            unique_list = list(unique_features)
            unique_descriptions = []
            for feat_idx in unique_list:
                feature = self.sae_model.features[feat_idx]
                desc = feature.name if hasattr(feature, 'name') and feature.name else f"Feature {feat_idx}"
                unique_descriptions.append(desc)
            
            comparison["unique_features"][word] = [
                {"feature": feat, "description": desc}
                for feat, desc in zip(unique_list, unique_descriptions)
            ]
        
        # Calculate feature overlap between pairs of words
        for i, word1 in enumerate(words):
            if word1 not in word_results:
                continue
                
            for word2 in words[i+1:]:
                if word2 not in word_results:
                    continue
                    
                features1 = set(word_results[word1]["top_feature_indices"])
                features2 = set(word_results[word2]["top_feature_indices"])
                
                overlap = features1 & features2
                overlap_list = list(overlap)
                
                # Get descriptions for overlapping features
                overlap_descriptions = []
                for feat_idx in overlap_list:
                    feature = self.sae_model.features[feat_idx]
                    desc = feature.name if hasattr(feature, 'name') and feature.name else f"Feature {feat_idx}"
                    overlap_descriptions.append(desc)
                
                pair_key = f"{word1}_{word2}"
                comparison["feature_overlap"][pair_key] = [
                    {"feature": feat, "description": desc}
                    for feat, desc in zip(overlap_list, overlap_descriptions)
                ]
        
        return comparison
    
    def visualize_comparison(self, comparison_results):
        """
        Visualize the comparison between multiple words.
        
        Args:
            comparison_results: Results from compare_words method
        """
        words = comparison_results["words"]
        
        if len(words) < 2:
            print("Need at least 2 words to visualize comparison")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(len(words), 1, figsize=(12, 4*len(words)), sharex=True)
        if len(words) == 1:
            axes = [axes]
        
        # Plot unique features for each word
        for i, word in enumerate(words):
            if word not in comparison_results["unique_features"]:
                continue
                
            unique_features = comparison_results["unique_features"][word]
            if not unique_features:
                axes[i].text(0.5, 0.5, f"No unique features for '{word}'", 
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Unique Features for '{word}'")
                continue
            
            feat_ids = [item["feature"] for item in unique_features]
            descriptions = [f"{feat}: {item['description'][:30]}..." 
                           if len(item['description']) > 30 else f"{feat}: {item['description']}"
                           for feat, item in zip(feat_ids, unique_features)]
            
            # Create bar chart
            axes[i].barh(range(len(feat_ids)), [1] * len(feat_ids), align='center')
            axes[i].set_yticks(range(len(feat_ids)))
            axes[i].set_yticklabels(descriptions)
            axes[i].set_title(f"Unique Features for '{word}'")
        
        plt.tight_layout()
        plt.show()
        
        # Plot feature overlaps for pairs
        if comparison_results["feature_overlap"]:
            # Determine number of pairs
            num_pairs = len(comparison_results["feature_overlap"])
            fig, axes = plt.subplots(num_pairs, 1, figsize=(12, 4*num_pairs), sharex=True)
            
            if num_pairs == 1:
                axes = [axes]
            
            for i, (pair_key, overlap) in enumerate(comparison_results["feature_overlap"].items()):
                words_pair = pair_key.split('_')
                
                if not overlap:
                    axes[i].text(0.5, 0.5, f"No overlapping features between '{words_pair[0]}' and '{words_pair[1]}'", 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f"Shared Features: '{words_pair[0]}' & '{words_pair[1]}'")
                    continue
                
                feat_ids = [item["feature"] for item in overlap]
                descriptions = [f"{feat}: {item['description'][:30]}..." 
                               if len(item['description']) > 30 else f"{feat}: {item['description']}"
                               for feat, item in zip(feat_ids, overlap)]
                
                # Create bar chart
                axes[i].barh(range(len(feat_ids)), [1] * len(feat_ids), align='center')
                axes[i].set_yticks(range(len(feat_ids)))
                axes[i].set_yticklabels(descriptions)
                axes[i].set_title(f"Shared Features: '{words_pair[0]}' & '{words_pair[1]}'")
            
            plt.tight_layout()
            plt.show()
    
    def __del__(self):
        """Clean up when object is deleted"""
        if self.hook_handle:
            self.hook_handle.remove()

# Example usage
def main():
    # Initialize analyzer for GPT2-small using layer 7
    analyzer = WordConceptAnalyzer(model_name="gpt2", layer=7)
    
    # Example 1: Analyze a single word
    word = "river"
    results = analyzer.analyze_word(word)
    print(f"\nAnalysis for '{word}':")
    print(f"Top feature activations: {results['top_feature_activations']}")
    print(f"Top feature indices: {results['top_feature_indices']}")
    print(f"Top feature descriptions:")
    for i, (idx, desc) in enumerate(zip(results['top_feature_indices'], results['top_feature_descriptions'])):
        print(f"  {i+1}. Feature {idx}: {desc}")
    
    # Get more info about the top feature
    top_feature = results['top_feature_indices'][0]
    feature_info = analyzer.get_feature_examples(top_feature)
    print(f"\nInformation about Feature {top_feature}:")
    print(f"  Name: {feature_info['name']}")
    print(f"  Description: {feature_info['description']}")
    print("  Examples:")
    for ex in feature_info['examples']:
        print(f"    - {ex}")
    
    # Visualize the results
    analyzer.visualize_word_concepts(word, results)
    
    # Example 2: Compare multiple words
    words = ["river", "ocean", "mountain"]
    comparison = analyzer.compare_words(words)
    
    # Display comparison results
    print("\nComparison results:")
    
    # Display shared features
    if comparison["shared_features"]:
        print(f"\nFeatures shared by all words ({', '.join(words)}):")
        for item in comparison["shared_features"]:
            print(f"  Feature {item['feature']}: {item['description']}")
    else:
        print(f"\nNo features shared by all words.")
    
    # Display word-specific unique features
    for word in words:
        if word in comparison["unique_features"]:
            unique = comparison["unique_features"][word]
            if unique:
                print(f"\nUnique features for '{word}':")
                for item in unique:
                    print(f"  Feature {item['feature']}: {item['description']}")
            else:
                print(f"\nNo unique features for '{word}'")
    
    # Display overlap between pairs
    for pair_key, overlap in comparison["feature_overlap"].items():
        word1, word2 = pair_key.split('_')
        if overlap:
            print(f"\nFeatures shared between '{word1}' and '{word2}':")
            for item in overlap:
                print(f"  Feature {item['feature']}: {item['description']}")
        else:
            print(f"\nNo features shared between '{word1}' and '{word2}'")
    
    # Visualize comparison
    analyzer.visualize_comparison(comparison)

if __name__ == "__main__":
    main()