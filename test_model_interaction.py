from model_interaction import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer

original_prompt = 'Customers browse supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'
perturbed_prompt = 'Customers graze supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'
random_prompt = 'Customers shop supermarket aisles selecting produce. Cart wheels squeaking. Cashier scanning barcodes efficiently. Shopping lists forgotten. Coupons saving money. Checkout lines moving slowly. Freezer section chilly. Shoppers comparing prices diligently. Expiration dates checked.'

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-small')
model = GPT2LMHeadModel.from_pretrained('gpt2-small')
model.eval()

# Generate outputs
original_output = generate_text(original_prompt, tokenizer, model)
perturbed_output = generate_text(perturbed_prompt, tokenizer, model)

print(original_output)
print(perturbed_output)
