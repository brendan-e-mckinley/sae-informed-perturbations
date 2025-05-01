import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from model_interaction import get_embeddings, cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

def parse_prompt_file(file_path):
    """
    Parse a file containing original, perturbed, and random prompts and their outputs.
    Returns a list of dictionaries, each containing the outputs for a set of prompts.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    results = {}

    # Split the content into sections based on the separator
    prompt_sets = content.split('----------------------------------------')
    
    results = []
    
    for prompt_set in prompt_sets:
        if not prompt_set.strip():
            continue
        
        # Initialize output variables
        original_output = ""
        random_output = ""
        
        # Split the set into sections (original, perturbed, random)
        sections = prompt_set.split('RANDOM_PROMPT_1:')
        
        # Extract the original output (everything after ORIGINAL_PROMPT: and before PERTURBED_PROMPT:)
        original_section = sections[0].replace('ORIGINAL_PROMPT:', '', 1).strip()
        original_output = original_section
        
        # Extract the perturbed output
        random_output = sections[1].strip()
        
        # Add to results
        results.append({
            'original_output': original_output,
            'random_output': random_output
        })
    
    return results

# Initialize variables
cos_sim_sae = []
cos_sim_random = []

# Read from file
prompt_outputs = parse_prompt_file("saved_output_distribution.txt")

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

for output in prompt_outputs:
    # Get embeddings
    original_embedding = get_embeddings(output['original_output'], tokenizer, model)
    random_embedding = get_embeddings(output['random_output'], tokenizer, model)
    
    # Calculate similarity metrics
    cos_sim_random.append(cosine_similarity(original_embedding, random_embedding)[0][0])

# Create a Gaussian KDE from the data
kde = gaussian_kde(cos_sim_random)

# Create a range of x values over which to evaluate the KDE
x_vals = np.linspace(.9, 1, 500)

# Evaluate the KDE over the x values
kde_vals = kde(x_vals)

df = pd.DataFrame({'x': x_vals, 'density': kde_vals})
df.to_csv('kde_curve.csv', index=False)