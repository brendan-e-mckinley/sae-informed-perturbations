from utils import getStrongestFeaturesForText, getWeakestTextForFeature
from model_interaction import compare_outputs
from nltk.corpus import wordnet

model_id = "gpt2-small"
source_set = "att-kk"
original_prompt = "quick brown fox jumps over the lazy dog"

out_of_vocabulary_word = "zzzzqqqq"

layer, indexes = getStrongestFeaturesForText(model_id, source_set, original_prompt)
saliency_prompts = []
for word in original_prompt.split():
    new_prompt = original_prompt.replace(word, out_of_vocabulary_word)
    saliency_prompts.append(new_prompt)

weakest_prompt_1 = getWeakestTextForFeature(model_id, layer, indexes[0], saliency_prompts, original_prompt)
weakest_prompt_2 = getWeakestTextForFeature(model_id, layer, indexes[1], saliency_prompts, original_prompt)
saliency_index = weakest_prompt_1.find(out_of_vocabulary_word)
target_word = original_prompt.split()[saliency_index]

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
weakest_synonym_prompt_1 = getWeakestTextForFeature(model_id, layer, indexes[0], synonym_prompts)
weakest_synonym_prompt_2 = getWeakestTextForFeature(model_id, layer, indexes[1], synonym_prompts)

print("Weakest synonym prompt for index {indexes[0]}: ", weakest_synonym_prompt_1)
print("Weakest synonym prompt for index {indexes[1]}: ", weakest_synonym_prompt_2)

original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt_1)
original_output, perturbed_output, metrics = compare_outputs(original_prompt, weakest_synonym_prompt_2)
