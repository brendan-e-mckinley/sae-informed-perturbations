from utils import getStrongestFeatureForText, getWeakestTextForFeature
from model_interaction import compare_outputs
from nltk.corpus import wordnet

model_id = "gpt2-small"
source_set = "res_scefr-ajt"
prompt = "The quick brown fox jumps over the lazy dog"

out_of_vocabulary_word = "zzzzqqqq"

layer, index = getStrongestFeatureForText(model_id, source_set, prompt)
saliency_prompts = []
for word in prompt.split():
    new_prompt = prompt.replace(word, out_of_vocabulary_word)
    saliency_prompts.append(new_prompt)

weakest_prompt = getWeakestTextForFeature(model_id, layer, index, saliency_prompts)
saliency_index = weakest_prompt.find(out_of_vocabulary_word)
target_word = prompt.split()[saliency_index]

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
            synonym_prompts.append(prompt.replace(target_word, word))


#layer, index = getStrongestFeatureForText(model_id, source_set, target_word)
weakest_synonym_prompt = getWeakestTextForFeature(model_id, layer, index, synonym_prompts)

print("Weakest synonym prompt: ", weakest_synonym_prompt)

original_output, perturbed_output, metrics = compare_outputs(prompt, weakest_synonym_prompt)
