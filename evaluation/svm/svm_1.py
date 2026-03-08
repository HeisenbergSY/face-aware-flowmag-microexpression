import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# Load features and labels from files
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Load group information (e.g., Subject + Filename + _x20) from the merged file
merged_data = pd.read_excel("CASME2-Merged.xlsx")
groups = merged_data.apply(lambda row: f"Cropped_sub{int(row['Subject']):02d}_{row['Filename']}_x20", axis=1).values

# Output file to save debug information
debug_file = "svm_debug_log.txt"

with open(debug_file, "w") as log:
    log.write("SVM Cross-Validation Debug Log\n")
    log.write("=" * 50 + "\n")

    # Perform Leave-One-Group-Out Cross-Validation (LOSO)
    logo = LeaveOneGroupOut()
    accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        log.write(f"\n=== Fold {fold_idx + 1} ===\n")

        # Train/Test groups
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        log.write(f"Train groups: {np.unique(train_groups)}\n")
        log.write(f"Test group: {np.unique(test_groups)}\n")

        # Train/Test labels
        train_labels = np.unique(y[train_idx])
        test_labels = np.unique(y[test_idx])
        log.write(f"Train labels: {train_labels}\n")
        log.write(f"Test labels: {test_labels}\n")

        # Train and test SVM
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        log.write(f"Accuracy for this fold: {acc:.2f}\n")
        accuracies.append(acc)

    # Write final average accuracy
    avg_accuracy = np.mean(accuracies)
    log.write("\n" + "=" * 50 + "\n")
    log.write(f"Average Accuracy: {avg_accuracy:.2f}\n")

print(f"Debug information saved to {debug_file}")
print(f"Average Accuracy: {avg_accuracy:.2f}\n")