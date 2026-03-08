# Data Setup

This repository uses the **CASME II** dataset for micro-expression analysis.

## Important

The dataset is **not included** in this repository.  
Users must obtain CASME II separately and place it in the expected folder structure.

## Expected Structure

```text
data/
├── raw/
│   └── CASME2/
├── processed/
└── masks/
```
## Folder Usage
- `raw/` — original CASME II data
- `processed/` — prepared frame sequences, resized data, or intermediate processed files
- `masks/` — facial landmark masks or other face-aware spatial priors used during training or evaluation
