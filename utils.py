import requests
import numpy

api_key = "sk-np-eXGRBzi2b8i6TAwF2mbv377Bf4K2hQHvq8tsXPO3ah00"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
activation_url = "https://www.neuronpedia.org/api/activation/new"
search_all_url = "https://www.neuronpedia.org/api/search-all"

def alignPrompts(shorter_tokens, longer_tokens, shorter_values, longer_values):
    shorter_indices_to_remove = []
    longer_indices_to_remove = []

    # shorter token index
    j = 0
    
    common_elements = list(set(shorter_tokens) & set(longer_tokens))

    for i in range(len(longer_tokens)):
        if longer_tokens[i] not in common_elements:
            longer_indices_to_remove.append(i)
    for j in range(len(shorter_tokens)):
        if shorter_tokens[j] not in common_elements:
            shorter_indices_to_remove.append(j)
    
    longer_values_aligned = longer_values.copy()
    mask = numpy.ones(len(longer_values_aligned), dtype=bool)
    mask[longer_indices_to_remove] = False
    longer_values_aligned = longer_values_aligned[mask]

    shorter_values_aligned = shorter_values.copy()
    mask = numpy.ones(len(shorter_values_aligned), dtype=bool)
    mask[shorter_indices_to_remove] = False
    shorter_values_aligned = shorter_values_aligned[mask]

    return shorter_values_aligned, longer_values_aligned

def getDiffForAlteredPrompt(model_id, layer, index, altered_prompt, values_original, tokens_original):
    payload_altered = {
        "feature": {
        "modelId": model_id,
        "source": layer,
        "index": index
    },
    "customText": altered_prompt
    }
    
    response_altered = requests.post(activation_url, headers=headers, json=payload_altered)
    # Handle the response
    if response_altered.status_code == 200:
        data_altered = response_altered.json()
        values_altered = numpy.array(data_altered.get("values", []))
        tokens_altered = numpy.array(data_altered.get("tokens", []))
    else:
        print("Request failed:", response_altered.status_code)
        print("Response content:", response_altered.content.decode())
        return

    if (len(tokens_altered) < len(tokens_original)):
        values_altered, values_original = alignPrompts(tokens_altered, tokens_original, values_altered, values_original)
    else:
        values_original, values_altered = alignPrompts(tokens_original, tokens_altered, values_original, values_altered)

    #diff = abs(numpy.linalg.norm(values_altered) - numpy.linalg.norm(values_original))
    diff = numpy.linalg.norm(values_altered - values_original)
    return diff

def getWeakestTextForFeature(model_id, layer, index, texts, original_prompt):
    # Initialize variables
    weakest_prompt = None
    greatest_diff = float(0)

    # Get activation values for original prompt
    payload_original = {
        "feature": {
        "modelId": model_id,
        "source": layer,
        "index": index
    },
    "customText": original_prompt
    }

    response_original = requests.post(activation_url, headers=headers, json=payload_original)
    
    # Handle the response
    if response_original.status_code == 200:
        data_original = response_original.json()
        values_original = numpy.array(data_original.get("values", []))
        tokens_original = numpy.array(data_original.get("tokens", []))
    else:
        print("Request failed:", response_original.status_code)
        print("Response content:", response_original.content.decode())

    for text in texts:
        # Get the diff for this text
        diff = getDiffForAlteredPrompt(model_id, layer, index, text, values_original, tokens_original)
        # Check if this is the largest diff so far
        if diff > greatest_diff:
            greatest_diff = diff
            weakest_prompt = text
    
    return weakest_prompt

def getStrongestFeaturesForText(model_id, source_set, target_word):

    payload = {
        "modelId": model_id,
        "sourceSet": source_set,
        "text": target_word,
        "selectedLayers": [],
        "sortIndexes": [1],         # Sort by importance
        "ignoreBos": False,
        "densityThreshold": -1,     # No threshold
        "numResults": 50            # Top 50 features
    }

    # Send the request
    response = requests.post(search_all_url, headers=headers, json=payload)

    # Handle the response
    if response.status_code == 200:
        data = response.json()
        results = data.get("result", [])
        indexes = []
        for i in range(2): 
            result = results[i]
            indexes.append(result['index'])
            selected_layer = result['layer']

        return selected_layer, indexes
    else:
        print("Request failed:", response.status_code)
        print("Response content:", response.content.decode())