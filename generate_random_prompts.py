from model_interaction import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import wordnet
import re
import random

def read_prompts_from_file(file_path):
    """
    Reads text prompts from a file and stores them in an array.
    
    Args:
        file_path (str): Path to the text file containing prompts
        
    Returns:
        list: Array of text prompts
    """
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip empty lines and lines that are just numbers followed by a period
                line = line.strip()
                if line and not line.strip().startswith('#'):
                    # Remove any line numbers (e.g., "1. ", "2. ", etc.)
                    if line[0].isdigit() and '. ' in line[:5]:
                        line = line[line.find('. ') + 2:]
                    prompts.append(line)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return prompts

def save_output_to_file(file, original_output, random_output_1):
    # Save to file
    try:
        file.write('ORIGINAL_PROMPT: ' + original_output + '\n')
        file.write('RANDOM_PROMPT_1: ' + random_output_1 + '\n')
        file.write('-' * 40 + '\n')
    except Exception as e:
        print(f"Error writing to file: {e}")

file_path = "final_prompts.txt"  # Change this to your file path
prompt_array = read_prompts_from_file(file_path)
    
# Print the number of prompts read
print(f"Successfully read {len(prompt_array)} prompts.")

model_id = "gpt2-small"
source_set = "att-kk"

for original_prompt in prompt_array:
    random_target_word = random.choice(original_prompt.split())
    random_synonyms = [ random_target_word ]
    random_synonym_prompts = []
    for random_synonym in wordnet.synsets(random_target_word):
        for random_lemma in random_synonym.lemmas():
            random_word = random_lemma.name()
            random_synonyms_lower = [random_text.lower() for random_text in random_synonyms]
            if random_word.lower() not in random_synonyms_lower:
                random_synonyms.append(random_word)
                random_pattern = r"\b" + re.escape(random_target_word) + r"\b"
                random_synonym_prompt = re.sub(random_pattern, random_word, original_prompt)
                random_synonym_prompts.append(random_synonym_prompt)
    
    if random_synonym_prompts == []:
        random_synonym_prompts.append(original_prompt) # Couldn't find any synonyms for randomly-selected target word

    random_target_synonym_prompt = random.choice(random_synonym_prompts)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Generate outputs
    original_output = generate_text(original_prompt, tokenizer, model)
    random_output_1 = generate_text(random_target_synonym_prompt, tokenizer, model)

    # Save to file
    file = open("random_random_prompts.txt", 'a', encoding='utf-8')
    save_output_to_file(file, original_output, random_output_1)
    file.close()
