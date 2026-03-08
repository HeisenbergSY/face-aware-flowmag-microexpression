import pandas as pd

# Load Excel files
coding_data = pd.read_excel("CASME2-coding-20140508.xlsx")
objective_classes_data = pd.read_excel("CASME2-ObjectiveClasses.xlsx")

# Merge the two files
merged_data = pd.merge(coding_data, objective_classes_data, on=["Subject", "Filename"])

# Save the merged file (optional)
merged_data.to_excel("CASME2-Merged.xlsx", index=False)
print("Merged data saved to CASME2-Merged.xlsx")
