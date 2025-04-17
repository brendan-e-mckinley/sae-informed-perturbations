import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock

# Function to get GPT-2 outputs
def generate_text(prompt, tokenizer, model, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to get embeddings from GPT-2's hidden states
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
    
    # Get the last hidden state (you could use other layers too)
    last_hidden_state = outputs.hidden_states[-1]
    
    # Use mean pooling to get a single vector for the whole sequence
    # We filter out padding tokens by using the attention mask
    mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
    mean_pooled = summed / counts
    
    return mean_pooled.numpy()

def compare_outputs(original_prompt, perturbed_prompt, metrics=True, visualize=True):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Generate outputs
    original_output = generate_text(original_prompt, tokenizer, model)
    perturbed_output = generate_text(perturbed_prompt, tokenizer, model)
    
    print("Original Output:", original_output[:100] + "..." if len(original_output) > 100 else original_output)
    print("Perturbed Output:", perturbed_output[:100] + "..." if len(perturbed_output) > 100 else perturbed_output)
    
    # Get embeddings
    original_embedding = get_embeddings(original_output, tokenizer, model)
    perturbed_embedding = get_embeddings(perturbed_output, tokenizer, model)
    
    results = {}
    
    if metrics:
        # Calculate similarity metrics
        cos_sim = cosine_similarity(original_embedding, perturbed_embedding)[0][0]
        euc_dist = euclidean(original_embedding.flatten(), perturbed_embedding.flatten())
        manhattan_dist = cityblock(original_embedding.flatten(), perturbed_embedding.flatten())
        
        # Normalize the distances relative to vector magnitudes
        norm_factor = np.linalg.norm(original_embedding) + np.linalg.norm(perturbed_embedding)
        norm_euc_dist = euc_dist / norm_factor
        norm_manhattan_dist = manhattan_dist / norm_factor
        
        results = {
            "cosine_similarity": cos_sim,
            "perturbation_impact_score": 1 - cos_sim,  # Higher means more different
            "euclidean_distance": euc_dist,
            "normalized_euclidean": norm_euc_dist,
            "manhattan_distance": manhattan_dist,
            "normalized_manhattan": norm_manhattan_dist
        }
        
        print("\nSimilarity Metrics:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    
    if visualize and original_embedding.shape[1] > 10:
        # Dimensionality reduction for visualization
        from sklearn.decomposition import PCA
        
        # Combine embeddings for PCA
        combined = np.vstack([original_embedding, perturbed_embedding])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', label='Original Output', s=100)
        plt.scatter(reduced[1, 0], reduced[1, 1], c='red', label='Perturbed Output', s=100)
        plt.arrow(reduced[0, 0], reduced[0, 1], 
                  reduced[1, 0]-reduced[0, 0], reduced[1, 1]-reduced[0, 1], 
                  head_width=0.01, head_length=0.01, fc='black', ec='black')
        
        plt.title('PCA Visualization of Embeddings')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    return original_output, perturbed_output, results