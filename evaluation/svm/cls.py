import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def preprocess_data():
    # Load data
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    merged_data = pd.read_excel("CASME2-Merged.xlsx")

    # Create mask for valid classes
    valid_classes = [1, 3, 4, 5, 7]  # Excluding classes 2 and 6
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    groups = merged_data["Subject"].values[mask]

    print(f"Dataset shape after filtering: {X.shape}")
    print("Class distribution:")
    for c in np.unique(y):
        print(f"Class {c}: {np.sum(y == c)} samples")

    return X, y, groups


def create_svm_pipeline(n_features=300):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(mutual_info_classif, k=n_features)),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="auto", class_weight="balanced"))
    ])


def train_and_evaluate():
    # Load and preprocess data
    X, y, groups = preprocess_data()

    # Initialize cross-validation
    logo = LeaveOneGroupOut()

    # Storage for results
    predictions = []
    true_labels = []
    feature_counts = [200, 300, 400]
    best_accuracy = 0
    best_n_features = 0

    # Open the file for saving results
    with open("svm_results.txt", "w") as f:
        f.write("SVM Training and Evaluation Results\n")
        f.write("=" * 50 + "\n")

        for n_features in feature_counts:
            f.write(f"\nTrying with {n_features} features...\n")
            print(f"\nTrying with {n_features} features...")
            curr_predictions = []
            curr_true_labels = []

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Create and train pipeline
                pipeline = create_svm_pipeline(n_features)
                pipeline.fit(X_train, y_train)

                # Predict
                y_pred = pipeline.predict(X_test)

                # Store predictions
                curr_predictions.extend(y_pred)
                curr_true_labels.extend(y_test)

            # Calculate accuracy
            curr_accuracy = accuracy_score(curr_true_labels, curr_predictions)
            f.write(f"Accuracy with {n_features} features: {curr_accuracy * 100:.2f}%\n")
            print(f"Accuracy with {n_features} features: {curr_accuracy * 100:.2f}%")

            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_n_features = n_features
                predictions = curr_predictions
                true_labels = curr_true_labels

        f.write(f"\nBest number of features: {best_n_features}\n")
        f.write(f"Best accuracy: {best_accuracy * 100:.2f}%\n")
        print(f"\nBest number of features: {best_n_features}")
        print(f"Best accuracy: {best_accuracy * 100:.2f}%")

        # Detailed classification report
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(true_labels, predictions))
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predictions))

    print("Results saved to 'svm_results.txt'")


if __name__ == "__main__":
    print("Starting enhanced SVM training...")
    train_and_evaluate()
