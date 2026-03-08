import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
from sklearn.model_selection import LeaveOneOut

# Load features, labels, and metadata
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
merged_data = pd.read_excel("CASME2-Merged.xlsx")

# Generate group information (based on Subject)
groups = merged_data["Subject"].values

# Scale the features
scaler = StandardScaler()
X_scaled = X

# Define the parameter grid
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

# Initialize Leave-One-Group-Out and SVM
logo = LeaveOneGroupOut()
svc = SVC(kernel="linear", C=10)  # Fix kernel and C value
loo = LeaveOneOut()

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=loo.split(X_scaled, y),
    scoring="accuracy",
    verbose=1,
    n_jobs=4
)

# Use the threading backend to avoid multiprocessing issues on Windows
with parallel_backend('threading', n_jobs=4):
    grid_search.fit(X_scaled, y)

# Print best parameters and accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validated Accuracy: {best_score:.2f}")

# Save the results for debugging
with open("svm_grid_search_scaled_results.txt", "w") as log:
    log.write(f"Best Parameters: {best_params}\n")
    log.write(f"Best Cross-Validated Accuracy: {best_score:.2f}\n")
