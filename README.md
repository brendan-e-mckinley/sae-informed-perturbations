# SAE-Informed Natural Language Adversarial Perturbations for LLMs

Brendan McKinley 

# Prompt Files

- `final_prompts.txt`: Contains a formatted list of sample prompts to evaluate various perturbation techniques against
- `final_distribution_prompts.txt`: Contains a formatted list of sample prompts to build a null distribution for the random perturbation technique

# Utility Files

These files contain the bulk of the implementation for the SAE-informed perturbation technique. 

- `utils.py`: Contains utility methods for determining the strongest features for a given text input and the weakest text for a given feature index
- `model_interaction.py`: Contains utility methods for interacting with GPT2, including generating text for a given input prompt and getting the embeddings for a given string.

# Test Files

- `test_multiple.py`: Reads the input prompts from `final_prompts.txt`, conducts the SAE-informed and random perturbation techniques for the most salient word for each prompt, and saves the results in `saved_output.txt`.
- `test_random.py`: Reads the input prompts from `final_prompts.txt`, conducs the random perturbation technique for a random word for each prompt, and saves the results in `saved_random_random_output.txt`.
- `test_iterative.py`: Iteratively perturbs a given `prompt` using the SAE-informed approach `perturbation_count` times.
- `test_model_interaction.py`: Evaluates the cosine similarity of the outputs for a given `original_prompt`, `perturbed_prompt`, and `random_prompt`. 

# KDE Files

- `compute_distribution.py`: Computes a KDE distribution for the cosine similarities for the text outputs stored in `saved_output_distribution.txt` and saves the distribution to `kde_curve.csv`. Input and output file names can be changed to save a distribution for the cosine similarities for the SAE-informed perturbation approach.
- `kde_curve_sae.csv`: The KDE distribution values for the SAE-informed perturbation approach.
- `kde_curve.csv`: The KDE distribution values for the null distribution. 

# Visualization Files

- `visualize_cosine_similarity_all.py`: Visualizes the cosine similarities for the SAE-informed and random perturbation techniques for the most salient word stored in `saved_output.txt` and the random perturbation technique for a random word stored in `saved_random_random_output.txt`. 
- `visualize_null_distribution.py`: Visualizes the KDE null distribution data saved in `kde_curve.csv` and the KDE distribution data for the SAE-informed perturbation approach saved in `kde_curve_sae.csv`. 
- `p_value.py`: Computes a p-value for the cosine similarities in the null distribution and those in the SAE-informed perturbation sample. 
