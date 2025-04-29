from model_interaction import generate_text, get_embeddings, cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

original_prompt = 'Customers browse supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'
perturbed_prompt = 'Customers graze supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'
random_prompt = 'Customers shop supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Generate outputs
original_output = generate_text(original_prompt, tokenizer, model)
perturbed_output = generate_text(perturbed_prompt, tokenizer, model)
random_output = generate_text(random_prompt, tokenizer, model)

# Get embeddings
original_embedding = get_embeddings(original_output, tokenizer, model)
perturbed_embedding = get_embeddings(perturbed_output, tokenizer, model)
random_embedding = get_embeddings(random_output, tokenizer, model)

# Calculate similarity metrics
cos_sim_sae = cosine_similarity(original_embedding, perturbed_embedding)[0][0]
cos_sim_random = cosine_similarity(original_embedding, random_embedding)[0][0]

x = 1

plt.plot(x, cos_sim_sae, marker='o', color='red')
plt.plot(x, cos_sim_random, marker='o', color='green')

# Add cosine similarity text
plt.figtext(0.5, 0.01, f'Cosine Similarity', ha='center', fontsize=12, 
            bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})

# Set equal aspect and grid
plt.grid(True)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Visualization')

plt.ylim(bottom=0.9990, top=1)

plt.show()

        