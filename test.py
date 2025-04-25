from utils import getStrongestFeaturesForText, getWeakestTextForFeature
from model_interaction import compare_outputs
from nltk.corpus import wordnet

model_id = "gpt2-small"
source_set = "att-kk"
#original_prompt = "movies and television are two of my favorite media. I really enjoy watching them"
original_prompt = "unfortunately lawyers will practice and engage in activites common to the legal profession"

layer, indexes = getStrongestFeaturesForText(model_id, source_set, original_prompt)
saliency_prompts = []
prompt_words = original_prompt.split()
prompt_words_no_duplicates = list(set(prompt_words)) # remove duplicate words
prompt_dict = {} # dictionary for tracking which saliency prompt is associated with which removed word
for index, word in enumerate(prompt_words_no_duplicates):
    new_prompt = original_prompt.replace(word, '')
    saliency_prompts.append(new_prompt)
    prompt_dict[index] = word

weakest_prompt = getWeakestTextForFeature(model_id, layer, indexes[0], saliency_prompts, original_prompt)
if weakest_prompt is None:
    weakest_prompt = getWeakestTextForFeature(model_id, layer, indexes[1], saliency_prompts, original_prompt)
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
            synonym_prompts.append(original_prompt.replace(target_word, word))


#layer, index = getStrongestFeaturesForText(model_id, source_set, target_word)
weakest_synonym_prompt = getWeakestTextForFeature(model_id, layer, indexes[0], synonym_prompts, original_prompt)
#weakest_synonym_prompt_2 = getWeakestTextForFeature(model_id, layer, indexes[1], synonym_prompts, original_prompt)

print("Weakest synonym prompt for index {indexes[0]}: ", weakest_synonym_prompt)
#print("Weakest synonym prompt for index {indexes[1]}: ", weakest_synonym_prompt_2)

original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt)
#original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt_2)
