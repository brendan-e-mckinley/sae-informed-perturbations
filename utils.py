import requests
import numpy

api_key = "sk-np-eXGRBzi2b8i6TAwF2mbv377Bf4K2hQHvq8tsXPO3ah00"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
layer = 9              # A mid-level layer where concept features tend to emerge

def getDiffForAlteredPrompt(model_id, layer, index, altered_prompt, original_prompt):
    # Request params for all requests
    url = "https://www.neuronpedia.org/api/activation/new"

    # Build the requests
    payload_original = {
        "feature": {
        "modelId": model_id,
        "source": layer,
        "index": index
    },
    "customText": original_prompt
    }
    payload_altered = {
        "feature": {
        "modelId": model_id,
        "source": layer,
        "index": index
    },
    "customText": altered_prompt
    }

    response_original = requests.post(url, headers=headers, json=payload_original)
    # Handle the response
    if response_original.status_code == 200:
        data_original = response_original.json()
        values_original = numpy.array(data_original.get("values", []))
    else:
        print("Request failed:", response_original.status_code)
        print("Response content:", response_original.content.decode())
        return
    
    response_altered = requests.post(url, headers=headers, json=payload_altered)
    # Handle the response
    if response_altered.status_code == 200:
        data_altered = response_altered.json()
        values_altered = numpy.array(data_altered.get("values", []))
    else:
        print("Request failed:", response_altered.status_code)
        print("Response content:", response_altered.content.decode())
        return
    
    diff = abs(numpy.linalg.norm(values_altered) - numpy.linalg.norm(values_original))
    return diff

def getWeakestTextForFeature(model_id, layer, index, texts, original_prompt):
    # Initialize variables
    weakest_prompt = None
    greatest_diff = float(0)

    for text in texts:
        # Get the diff for this text
        diff = getDiffForAlteredPrompt(model_id, layer, index, text, original_prompt)
        # Check if this is the largest diff so far
        if diff > greatest_diff:
            greatest_diff = diff
            weakest_prompt = text
    
    return weakest_prompt

def getStrongestFeaturesForText(model_id, source_set, target_word):
    # Compose the layer ID string expected by the API
    #selected_layer = f"{layer}-{source_set}"

    # Build the request
    url = "https://www.neuronpedia.org/api/search-all"
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
    response = requests.post(url, headers=headers, json=payload)

    # Handle the response
    if response.status_code == 200:
        data = response.json()
        results = data.get("result", [])
        indexes = []
        for i in range(2): 
            result = results[i]
            indexes.append(result['index'])
            selected_layer = result['layer']
        # print(f"Top concept features for the input: '{target_word}'\n")
        # for feature in results:
        #     index = feature['index']
        #     print(f"Index: {feature['index']}, Max Value: {feature['maxValue']}")

        return selected_layer, indexes
    else:
        print("Request failed:", response.status_code)
        print("Response content:", response.content.decode())

    # Build a new request to understand concept feature
    # url =  "https://www.neuronpedia.org/api/vector/get"
    # headers = {
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "modelId": model_id,
    #     "source": top_layer,
    #     "index": top_index
    # }

    # response = requests.post(url, headers=headers, json=payload)
    # if response.status_code == 200:
    #     data = response.json()
    #     result = data.get("hookName")
    #     hook_name = result.get("hookName")
    #     print(f"Top Concept Feature Hook Name: {hook_name}")
    # else:
    #     print("Request failed:", response.status_code)
    #     print("Response content:", response.content.decode())