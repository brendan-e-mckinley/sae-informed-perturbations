import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the KDE data from CSV
df_random = pd.read_csv('kde_curve.csv')

# Extract x and density values
x_vals_random = df_random['x'].values
kde_vals_random = df_random['density'].values

# Plot the KDE curve
plt.plot(x_vals_random, kde_vals_random, label='Null distribution, random perturbations', color='blue')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.title('KDE Distribution')
plt.legend()
plt.grid(True)

# Load the KDE data from CSV
df_sae = pd.read_csv('kde_curve_sae.csv')

# Extract x and density values
x_vals_sae = df_sae['x'].values
kde_vals_sae = df_sae['density'].values

# Plot the KDE curve
plt.plot(x_vals_sae, kde_vals_sae, label='Distribution, SAE-informed perturbations', color='red')
plt.legend()
plt.show()