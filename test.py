from utils import getStrongestFeatureForWord, getWeakestWordForFeature

model_id = "gpt2-small"
target_word = "upset"
syonyms = ["trouble",
"perturb",
"discompose",
"unsettle",
"disconcert",
"distress",
"discountenance",
"dismay",
"disquiet",
"worry",
"bother",
"plague"]

layer, index = getStrongestFeatureForWord(model_id, target_word)
weakest_synonym = getWeakestWordForFeature(model_id, layer, index, syonyms)

print("Weakest synonym: ", weakest_synonym)
