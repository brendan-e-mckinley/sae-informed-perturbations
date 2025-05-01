
import matplotlib.pyplot as plt
import numpy as np
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
    
    # Split the content by the separator
    prompt_sets = content.split('----------------------------------------')
    
    results = []
    
    for prompt_set in prompt_sets:
        if not prompt_set.strip():
            continue
        
        # Initialize output variables
        original_output = ""
        perturbed_output = ""
        random_output = ""
        
        # Split the set into sections (original, perturbed, random)
        sections = prompt_set.split('RANDOM_PROMPT:')
            
        # Process the first part which contains ORIGINAL and PERTURBED
        first_sections = sections[0].split('PERTURBED_PROMPT:')
            
        # Extract the original output (everything after ORIGINAL_PROMPT: and before PERTURBED_PROMPT:)
        original_section = first_sections[0].replace('ORIGINAL_PROMPT:', '', 1).strip()
        original_output = original_section
        
        # Extract the perturbed output
        perturbed_output = first_sections[1].strip()
        
        # Extract the random output
        random_output = sections[1].strip()
        
        # Add to results
        results.append({
            'original_output': original_output,
            'perturbed_output': perturbed_output,
            'random_output': random_output
        })
    
    return results

def parse_prompt_file_random(file_path):
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
cos_sim_random = []

# Read from file
prompt_outputs_random = parse_prompt_file_random("generated_output_distribution.txt")

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

for output in prompt_outputs_random:
    # Get embeddings
    original_embedding = get_embeddings(output['original_output'], tokenizer, model)
    random_random_embedding = get_embeddings(output['random_output'], tokenizer, model)
    
    # Calculate similarity metrics
    cos_sim_random.append(cosine_similarity(original_embedding, random_random_embedding)[0][0])

# Initialize variables
cos_sim_sae = []

# Read from file
prompt_outputs = parse_prompt_file("saved_output.txt")

for output in prompt_outputs:
    # Get embeddings
    original_embedding = get_embeddings(output['original_output'], tokenizer, model)
    perturbed_embedding = get_embeddings(output['perturbed_output'], tokenizer, model)
    
    # Calculate similarity metrics
    cos_sim_sae.append(cosine_similarity(original_embedding, perturbed_embedding)[0][0])

mu_sae = np.mean(cos_sim_sae)
mu_null = np.mean(cos_sim_random)

test_statistic = mu_sae - mu_null

print('Average SAE: ', mu_sae)
print('Average Random: ', mu_null)
print('Test statistic: ', test_statistic)

# Step 3: Combine the data
combined = np.concatenate([cos_sim_random, cos_sim_sae])
n_random = len(cos_sim_random)
n_special = len(cos_sim_sae)

# Step 4: Run permutation test
n_iterations = 10000
diffs = np.zeros(n_iterations)

for i in range(n_iterations):
    np.random.shuffle(combined)
    group_random = combined[:n_random]
    group_special = combined[n_random:]
    diffs[i] = np.mean(group_random) - np.mean(group_special)

# Step 5: Compute p-value (two-tailed)
p_value = np.mean(np.abs(diffs) >= np.abs(test_statistic))

print(f"Observed difference in means: {test_statistic:.4f}")
print(f"Empirical two-tailed p-value: {p_value:.4f}")
