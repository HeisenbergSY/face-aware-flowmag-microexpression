# Evaluation

This folder contains scripts, notes, and results related to evaluating motion magnification and downstream micro-expression analysis.

## Purpose

The evaluation pipeline is intended to support comparison between:

- pretrained baseline inference
- test-time adaptation
- face-aware fine-tuning variants
- different landmark regularization strengths

## Possible Evaluation Components

- qualitative inspection of magnified motion
- optical flow consistency analysis
- LBP-TOP feature extraction
- SVM-based classification
- comparison of different experimental settings

## Suggested Structure

```text
evaluation/
├── matlab/
├── lbptop/
├── svm/
└── results/