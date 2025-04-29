
import matplotlib.pyplot as plt
from model_interaction import get_embeddings, cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        
        if len(sections) < 2:
            continue
            
        # Process the first part which contains ORIGINAL and PERTURBED
        first_sections = sections[0].split('PERTURBED_PROMPT:')
        
        if len(first_sections) < 2:
            continue
            
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

# Initialize variables
cos_sim_sae = []
cos_sim_random = []

# TO DO: Read from file
prompt_outputs = parse_prompt_file("saved_output.txt")

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Get embeddings
original_embedding = get_embeddings(original_output, tokenizer, model)
perturbed_embedding = get_embeddings(perturbed_output, tokenizer, model)
random_embedding = get_embeddings(random_output, tokenizer, model)

# Calculate similarity metrics
cos_sim_sae.append(cosine_similarity(original_embedding, perturbed_embedding)[0][0])
cos_sim_random.append(cosine_similarity(original_embedding, random_embedding)[0][0])

plt.plot(range(100), cos_sim_sae, linestyle='--', marker='o', color='red')
plt.plot(range(100), cos_sim_random, linestyle='--', marker='o', color='green')

# Add cosine similarity text
plt.figtext(0.5, 0.01, f'Cosine Similarity', ha='center', fontsize=12, 
            bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})

# Set equal aspect and grid
plt.grid(True)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Visualization')

plt.show()