import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# Load the earthquake data (replace 'your_data_file.csv' with the actual data file)
data = np.genfromtxt('/Users/kalyanitidke/Downloads/fijiquakes_data.csv', delimiter=',')

# Step 1: Estimate the empirical CDF
ecdf = ECDF(data)

# Step 2: Compute and plot a 95% confidence envelope for F(x)
n = len(data)
x_values = np.sort(data)
lower_confidence = ecdf(x_values) - 1.96 / np.sqrt(n)
upper_confidence = ecdf(x_values) + 1.96 / np.sqrt(n)

plt.figure(figsize=(10, 6))
plt.plot(x_values, ecdf(x_values), label='Empirical CDF', color='black')
plt.step(x_values, upper_confidence, linestyle='dashed', color='red', label='95% Confidence Envelope')
plt.step(x_values, lower_confidence, linestyle='dashed', color='red')
plt.fill_between(x_values, upper_confidence, lower_confidence, color='red', alpha=0.2)
plt.xlabel('Magnitude of Earthquakes')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF with 95% Confidence Envelope')
plt.legend()
plt.show()

# Step 3: Compute and print the 95% confidence interval for F(4.9) - F(4.3)
F_49 = ecdf(4.9)
F_43 = ecdf(4.3)
confidence_interval = (F_49 - F_43) - 1.96 * np.sqrt((F_49 * (1 - F_49) / n) + (F_43 * (1 - F_43) / n))
print(f"95% Confidence Interval for F(4.9) - F(4.3): {confidence_interval}")
