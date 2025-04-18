from utils import getStrongestFeatureForWord, getWeakestWordForFeature
from model_interaction import compare_outputs
from nltk.corpus import wordnet

model_id = "gpt2-small"
prompt = "The quick brown fox jumps over the lazy dog"

target_word = "quick" # TO DO: compute word saliency scores for each word in prompt and choose most important word

synonyms = [ target_word ]
for synonym in wordnet.synsets(target_word):
    for lemma in synonym.lemmas():
        word = lemma.name()
        if word not in synonyms:
            synonyms.append(word)

layer, index = getStrongestFeatureForWord(model_id, target_word)
weakest_synonym = getWeakestWordForFeature(model_id, layer, index, synonyms)

modified_prompt = prompt.replace(target_word, weakest_synonym)

print("Weakest synonym: ", weakest_synonym)

original_output, perturbed_output, metrics = compare_outputs(prompt, modified_prompt)
