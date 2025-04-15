import requests

def getWeakestWordForFeature(model_id, layer, index, synonyms):
    # Initialize variables
    weakest_word = None
    smallest_max_value = float('inf')

    # Request params for all requests
    url = "https://www.neuronpedia.org/api/activation/new"
    headers = {
        "Content-Type": "application/json"
    }

    for synonym in synonyms:
      # Build the request
      payload = {
          "feature": {
          "modelId": model_id,
          "source": layer,
          "index": index
        },
        "customText": synonym
      }

      response = requests.post(url, headers=headers, json=payload)

      # Handle the response
      if response.status_code == 200:
          data = response.json()
          max_value = data.get("maxValue", [])

          print(synonym + " value: ", max_value)

          # Check if this is the smallest max_value so far
          if max_value is not None and max_value < smallest_max_value:
              smallest_max_value = max_value
              weakest_word = synonym
      else:
          print("Request failed:", response.status_code)
          print("Response content:", response.content.decode())

    return weakest_word

def getStrongestFeatureForWord(model_id, target_word):
    source_set = "res-jb"  # Pretrained SAE set
    layer = 6              # A mid-level layer where concept features tend to emerge

    # Compose the layer ID string expected by the API
    selected_layer = f"{layer}-res-jb"

    # Build the request
    url = "https://www.neuronpedia.org/api/search-all"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "modelId": model_id,
        "sourceSet": source_set,
        "text": target_word,
        "selectedLayers": [selected_layer],
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
        top_result = results[0]
        top_layer = top_result['layer']
        top_index = top_result['index']
        # print(f"Top concept features for the input: '{target_word}'\n")
        # for feature in results:
        #     index = feature['index']
        #     print(f"Index: {feature['index']}, Max Value: {feature['maxValue']}")

        return top_layer, top_index
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