from utils import getStrongestFeaturesForText, getWeakestTextForFeature
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

def save_output_to_file(file, original_output, saliency_output, random_output):
    # Save to file
    try:
        file.write('ORIGINAL_PROMPT: ' + original_output + '\n')
        file.write('SALIENCY_PROMPT: ' + saliency_output + '\n')
        file.write('RANDOM_PROMPT: ' + random_output + '\n')
        file.write('-' * 40 + '\n')
    except Exception as e:
        print(f"Error writing to file: {e}")

file_path = "prompts.txt"  # Change this to your file path
prompt_array = read_prompts_from_file(file_path)
    
# Print the number of prompts read
print(f"Successfully read {len(prompt_array)} prompts.")

model_id = "gpt2-small"
source_set = "att-kk"

for original_prompt in prompt_array:
    layer, indexes = getStrongestFeaturesForText(model_id, source_set, original_prompt)
    saliency_prompts = []
    prompt_words = original_prompt.split()
    prompt_words_no_duplicates = list(set(prompt_words)) # remove duplicate words
    prompt_dict = {} # dictionary for tracking which saliency prompt is associated with which removed word
    for index, word in enumerate(prompt_words_no_duplicates):
        pattern = r"\b" + re.escape(word) + r"\b"
        new_prompt = re.sub(pattern, '', original_prompt)
        saliency_prompts.append(new_prompt)
        prompt_dict[index] = word

    weakest_prompt = getWeakestTextForFeature(model_id, layer, indexes[0], saliency_prompts, original_prompt)
    index = indexes[0]
    if weakest_prompt is None:
        weakest_prompt = getWeakestTextForFeature(model_id, layer, indexes[1], saliency_prompts, original_prompt)
        index = indexes[1]

    saliency_index = saliency_prompts.index(weakest_prompt)
    target_word = prompt_dict[saliency_index]

    synonyms = [ target_word ]
    synonym_prompts = []
    for synonym in wordnet.synsets(target_word):
        for lemma in synonym.lemmas():
            word = lemma.name()
            synonyms_lower = [text.lower() for text in synonyms]
            if word.lower() not in synonyms_lower:
                synonyms.append(word)
                pattern = r"\b" + re.escape(target_word) + r"\b"
                synonym_prompt = re.sub(pattern, word, original_prompt)
                synonym_prompts.append(synonym_prompt)

    target_synonym_prompt = random.choice(synonym_prompts)

    random_target_word = random.choice(original_prompt.split())
    random_synonyms = [ random_target_word ]
    random_synonym_prompts = []
    for random_synonym in wordnet.synsets(random_target_word):
        for random_lemma in random_synonym.lemmas():
            random_word = random_lemma.name()
            random_synonyms_lower = [random_text.lower() for random_text in random_synonyms]
            if random_word.lower() not in random_synonyms_lower:
                random_synonyms.append(word)
                random_pattern = r"\b" + re.escape(random_target_word) + r"\b"
                random_synonym_prompt = re.sub(random_pattern, random_word, original_prompt)
                random_synonym_prompts.append(random_synonym_prompt)
    
    random_target_synonym_prompt = random.choice(random_synonym_prompts)

    print(f"Weakest synonym prompt for index {index}: ", target_synonym_prompt)
    print(f"Weakest synonym prompt for index {index}: ", random_target_synonym_prompt)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Generate outputs
    original_output = generate_text(original_prompt, tokenizer, model)
    saliency_output = generate_text(target_synonym_prompt, tokenizer, model)
    random_output = generate_text(random_target_synonym_prompt, tokenizer, model)

    # Save to file
    file = open("generated_output.txt", 'a', encoding='utf-8')
    save_output_to_file(file, original_output, saliency_output, random_output)
    file.close()
