from utils import getStrongestFeaturesForText, getWeakestTextForFeature
from model_interaction import compare_outputs
from nltk.corpus import wordnet
import re

model_id = "gpt2-small"
source_set = "att-kk"
#original_prompt = "movies and television are two of my favorite media. I really enjoy watching them"
original_prompt = "Customers browse supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked."

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

#target_word = "quick" # TO DO: compute word saliency scores for each word in prompt and choose most important word

# synonyms = [ target_word ]
# for synonym in wordnet.synsets(target_word):
#     for lemma in synonym.lemmas():
#         word = lemma.name()
#         if word not in synonyms:
#             synonyms.append(word)

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


#layer, index = getStrongestFeaturesForText(model_id, source_set, target_word)
weakest_synonym_prompt = getWeakestTextForFeature(model_id, layer, index, synonym_prompts, original_prompt)
#weakest_synonym_prompt_2 = getWeakestTextForFeature(model_id, layer, indexes[1], synonym_prompts, original_prompt)

print("Weakest synonym prompt for index {index}: ", weakest_synonym_prompt)
#print("Weakest synonym prompt for index {indexes[1]}: ", weakest_synonym_prompt_2)

original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt)
#original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt_2)
