import torch

# Function to get GPT-2 outputs
def generate_text(prompt, tokenizer, model, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to get embeddings from GPT-2's hidden states
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
    
    # Get the last hidden state (you could use other layers too)
    last_hidden_state = outputs.hidden_states[-1]
    
    # Use mean pooling to get a single vector for the whole sequence
    # We filter out padding tokens by using the attention mask
    mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
    mean_pooled = summed / counts
    
    return mean_pooled.numpy()