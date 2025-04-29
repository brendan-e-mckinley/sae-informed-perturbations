from utils import getStrongestFeaturesForText, getWeakestTextForFeature
from model_interaction import generate_text, get_embeddings, cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import wordnet

import matplotlib.pyplot as plt
import re
import random

model_id = "gpt2-small"
source_set = "att-kk"
prompt = "Customers browse supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked."
original_prompt = prompt

perturbation_index = list(range(5))
cos_sim_sae = []
cos_sim_random = []

for index in perturbation_index:
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
            if word not in synonyms:
                synonyms.append(word)
                pattern = r"\b" + re.escape(target_word) + r"\b"
                synonym_prompt = re.sub(pattern, word, original_prompt)
                synonym_prompts.append(synonym_prompt)

    random_synonym_prompt = random.choice(synonym_prompts)

    weakest_synonym_prompt = getWeakestTextForFeature(model_id, layer, index, synonym_prompts, original_prompt)

    print("Weakest synonym prompt for index {index}: ", weakest_synonym_prompt)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Generate outputs
    original_output = generate_text(original_prompt, tokenizer, model)
    perturbed_output = generate_text(weakest_synonym_prompt, tokenizer, model)
    random_output = generate_text(random_synonym_prompt, tokenizer, model)

    # Get embeddings
    original_embedding = get_embeddings(original_output, tokenizer, model)
    perturbed_embedding = get_embeddings(perturbed_output, tokenizer, model)
    random_embedding = get_embeddings(random_output, tokenizer, model)

    # Calculate similarity metrics
    cos_sim_sae.append(cosine_similarity(original_embedding, perturbed_embedding)[0][0])
    cos_sim_random.append(cosine_similarity(original_embedding, random_embedding)[0][0])

    original_prompt = weakest_synonym_prompt

plt.plot(perturbation_index, cos_sim_sae, linestyle='--', marker='o', color='red')
plt.plot(perturbation_index, cos_sim_random, linestyle='--', marker='o', color='green')

# Add cosine similarity text
plt.figtext(0.5, 0.01, f'Cosine Similarity', ha='center', fontsize=12, 
            bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})

# Set equal aspect and grid
plt.grid(True)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Visualization')

plt.ylim(bottom=0.9990, top=1)

plt.show()
