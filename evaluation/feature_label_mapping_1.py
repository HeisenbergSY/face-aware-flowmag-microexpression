import os
import numpy as np
import pandas as pd

# Load the merged Excel file
merged_data = pd.read_excel("CASME2-Merged.xlsx")

# Directory where features are stored
data_dir = "Cropped\Cropped"

def load_features_and_labels(data_dir, merged_data):
    X = []  # Features
    y = []  # Labels (Objective Classes)

    for _, row in merged_data.iterrows():
        subject = int(row["Subject"])  # Subject ID
        filename = row["Filename"]    # Filename
        objective_class = row["Objective Class"]  # Label

        # Construct the folder name
        folder_name = f"Cropped_sub{subject:02d}_{filename}_x20"
        feature_path = os.path.join(data_dir, folder_name, "lbp_top_features.npy")

        # Check if the file exists
        if os.path.exists(feature_path):
            # Load features and store them with the corresponding label
            X.append(np.load(feature_path))
            y.append(objective_class)
        else:
            print(f"Feature file not found: {feature_path}")

    return np.array(X), np.array(y)

# Directory where the folders are located
data_dir = "Cropped\Cropped"

# Load the features and labels
X, y = load_features_and_labels(data_dir, merged_data)
print(f"Loaded {len(X)} features and {len(y)} labels.")


# Output results for debugging
print(f"Loaded {len(X)} features and {len(y)} labels.")
if len(X) > 0:
    print(f"Shape of first feature vector: {X[0].shape}")

if len(X) > 0 and len(y) > 0:
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("Features and labels saved to X_features.npy and y_labels.npy.")
else:
    print("No features or labels to save.")
print(f"First feature vector shape: {X[0].shape}")
print(f"First label: {y[0]}")