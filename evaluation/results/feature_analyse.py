import numpy as np
import matplotlib.pyplot as plt

# Load the features and labels
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Check for NaNs or Zeros
nan_counts = np.isnan(X).sum(axis=1)
zero_counts = (X == 0).sum(axis=1)

# Summary statistics
mean_values = np.mean(X, axis=1)
std_values = np.std(X, axis=1)

# Log the results
print("Feature Quality Analysis:")
print("=" * 50)
print(f"Number of feature vectors: {len(X)}")
print(f"Feature vector dimensionality: {X[0].shape[0]}")
print(f"Feature vectors with NaNs: {np.sum(nan_counts > 0)}")
print(f"Feature vectors with all zeros: {np.sum(zero_counts == X.shape[1])}")
print("=" * 50)

# Plot distributions
plt.figure(figsize=(12, 8))

# Mean distribution
plt.subplot(2, 2, 1)
plt.hist(mean_values, bins=30, color="blue", alpha=0.7)
plt.title("Mean Values Distribution")
plt.xlabel("Mean")
plt.ylabel("Frequency")

# Standard deviation distribution
plt.subplot(2, 2, 2)
plt.hist(std_values, bins=30, color="green", alpha=0.7)
plt.title("Standard Deviation Distribution")
plt.xlabel("Standard Deviation")
plt.ylabel("Frequency")

# NaN counts
plt.subplot(2, 2, 3)
plt.hist(nan_counts, bins=30, color="red", alpha=0.7)
plt.title("NaN Counts per Feature Vector")
plt.xlabel("NaN Count")
plt.ylabel("Frequency")

# Zero counts
plt.subplot(2, 2, 4)
plt.hist(zero_counts, bins=30, color="orange", alpha=0.7)
plt.title("Zero Counts per Feature Vector")
plt.xlabel("Zero Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Save results for debugging
np.savetxt("feature_quality_analysis.txt", np.column_stack([mean_values, std_values, nan_counts, zero_counts]),
           header="Mean, Std, NaN Counts, Zero Counts")
print("Feature quality analysis saved to feature_quality_analysis.txt.")
_