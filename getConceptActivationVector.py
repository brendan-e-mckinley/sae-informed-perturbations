import requests
import json

# Define your input
target_word = "cat"
input_text = f"The animal is a {target_word}."
model_id = "gpt2-small"
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
    "text": input_text,
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
    print(f"Top concept features for the input: '{input_text}'\n")
    for feature in data.get("result", []):
        print(f"Index: {feature['index']}, Max Value: {feature['maxValue']}")
else:
    print("Request failed:", response.status_code)
    print("Response content:", response.content.decode())