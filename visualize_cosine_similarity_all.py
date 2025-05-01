
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
cos_sim_random_random = []

# Read from file
prompt_outputs_random = parse_prompt_file_random("saved_random_random_output.txt")

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

for output in prompt_outputs_random:
    # Get embeddings
    original_embedding = get_embeddings(output['original_output'], tokenizer, model)
    random_random_embedding = get_embeddings(output['random_output'], tokenizer, model)
    
    # Calculate similarity metrics
    cos_sim_random_random.append(cosine_similarity(original_embedding, random_random_embedding)[0][0])

# Initialize variables
cos_sim_sae = []
cos_sim_random = []

# Read from file
prompt_outputs = parse_prompt_file("saved_output.txt")

for output in prompt_outputs:
    # Get embeddings
    original_embedding = get_embeddings(output['original_output'], tokenizer, model)
    perturbed_embedding = get_embeddings(output['perturbed_output'], tokenizer, model)
    random_embedding = get_embeddings(output['random_output'], tokenizer, model)
    
    # Calculate similarity metrics
    cos_sim_sae.append(cosine_similarity(original_embedding, perturbed_embedding)[0][0])
    cos_sim_random.append(cosine_similarity(original_embedding, random_embedding)[0][0])

plt.plot(range(80), cos_sim_sae, linestyle='--', marker='o', color='red', label='SAE-Informed Perturbation of Most Salient Word')
plt.plot(range(80), cos_sim_random, linestyle='--', marker='o', color='green', label='Random Perturbation of Most Salient Word')
plt.plot(range(80), cos_sim_random_random, linestyle='--', marker='o', color='blue', label='Random Perturbation of Random Word')

# Add cosine similarity text
plt.figtext(0.5, 0.01, f'Cosine Similarity', ha='center', fontsize=12, 
            bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})

# Set equal aspect and grid
plt.grid(True)

# Add labels and title
plt.xlabel('Prompt Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Visualization')

plt.legend()
plt.show()

# Groups 
sae_groups = []
random_groups = []
random_random_groups = []

group_count = np.arange(19, 80, 20)

for count in group_count:
    sae_groups.append(sum(cos_sim_sae[(count - 19):count])/19)
    random_groups.append(sum(cos_sim_random[(count - 19):count])/19)
    random_random_groups.append(sum(cos_sim_random_random[(count - 19):count])/19)

    print("SAE group: ", sum(cos_sim_sae[(count - 19):count])/19)
    print("Random group: ", sum(cos_sim_random[(count - 19):count])/19)
    print("Random random group: ", sum(cos_sim_random_random[(count - 19):count])/19)
    

plt.figure()

plt.plot(group_count, sae_groups, linestyle='--', marker='o', color='red', label='SAE-Informed Perturbation of Most Salient Word')
plt.plot(group_count, random_groups, linestyle='--', marker='o', color='green', label='Random Perturbation of Most Salient Word')
plt.plot(group_count, random_random_groups, linestyle='--', marker='o', color='blue', label='Random Perturbation of Random Word')

# Add cosine similarity text
plt.figtext(0.5, 0.01, f'Average Cosine Similarity by Group Size', ha='center', fontsize=12, 
            bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})

# Set equal aspect and grid
plt.grid(True)

# Add labels and title
plt.xlabel('Group Size')
plt.ylabel('Average Cosine Similarity')
plt.title('Average Cosine Similarity Visualization')

plt.legend()
plt.show()