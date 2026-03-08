import os
import numpy as np
import pandas as pd

# Load the merged Excel file
merged_data = pd.read_excel("CASME2-Merged.xlsx")

# Directory where features are stored
data_dir = r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped"  # Update to your feature directory

# Debug log file
debug_log_file = "feature_label_mapping_debug_log.txt"

def load_features_and_labels(data_dir, merged_data, debug_log_file):
    X = []  # Features
    y = []  # Labels (Objective Classes)

    with open(debug_log_file, "w") as log:
        log.write("Feature-Label Mapping Debug Log\n")
        log.write("=" * 80 + "\n")
        
        for idx, row in merged_data.iterrows():
            subject = int(row["Subject"])  # Subject ID
            filename = row["Filename"]    # Filename
            objective_class = row["Objective Class"]  # Label

            # Construct the folder path
            folder_path = os.path.join(data_dir, f"sub{subject:02d}", filename)
            feature_path = os.path.join(folder_path, "lbp_top_features.npy")

            # Log the current sample details
            log.write(f"Row {idx + 1}:\n")
            log.write(f"  Subject: {subject}\n")
            log.write(f"  Filename: {filename}\n")
            log.write(f"  Objective Class: {objective_class}\n")
            log.write(f"  Expected Feature Path: {feature_path}\n")

            # Check if the file exists
            if os.path.exists(feature_path):
                # Load features and store them with the corresponding label
                features = np.load(feature_path)
                X.append(features)
                y.append(objective_class)

                log.write(f"  Feature Vector Shape: {features.shape}\n")
                log.write(f"  Feature Vector Example: {features[:5]}...\n")  # Log first 5 values
            else:
                log.write("  Feature file not found.\n")

            log.write("-" * 80 + "\n")
    
    return np.array(X), np.array(y)

# Load features and labels with debug logging
X, y = load_features_and_labels(data_dir, merged_data, debug_log_file)

# Output results for debugging
print(f"Loaded {len(X)} features and {len(y)} labels.")
if len(X) > 0:
    print(f"Shape of first feature vector: {X[0].shape}")
    print(f"First label: {y[0]}")

if len(X) > 0 and len(y) > 0:
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("Features and labels saved to X_features.npy and y_labels.npy.")
else:
    print("No features or labels to save.")
print(f"First feature vector shape: {X[0].shape}")
print(f"First label: {y[0]}")