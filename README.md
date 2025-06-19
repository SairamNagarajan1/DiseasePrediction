# Disease Prediction Using Machine Learning

This repository contains a machine learning project for predicting diseases based on symptom inputs. The project implements a comprehensive pipeline that includes data preprocessing, class balancing, training of multiple machine learning models (SVM, Naive Bayes, Random Forest), ensemble prediction using majority voting, and visualization of results through confusion matrices. The code is designed to work with the `improved_disease_dataset.csv` dataset and includes a function to predict diseases from user-provided symptoms.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Results](#results)
- [Contact](#contact)

## Features
- Loads and preprocesses the disease dataset with label encoding and missing value handling.
- Balances class distribution using RandomOverSampler to address imbalanced data.
- Trains and evaluates three machine learning models: Support Vector Machine (SVM), Naive Bayes, and Random Forest.
- Combines model predictions using a majority voting ensemble for improved robustness.
- Visualizes class distribution and confusion matrices using Matplotlib and Seaborn.
- Provides a prediction function to diagnose diseases based on input symptoms (e.g., "Itching,Skin Rash,Nodal Skin Eruptions").

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Notes for Customization
1. **Repository URL**: The URL `https://github.com/SairamNagarajan1/DiseasePrediction` is used as requested. Ensure this matches your actual repository.
2. **File Name**: The script is named `disease_prediction.py` for consistency. If your file has a different name (e.g., based on the repository), update the usage section accordingly.
3. **Dataset Path**: The README assumes the dataset is in `/content/`. Adjust if your setup differs (e.g., local directory).
4. **Author Details**: Replace the placeholder email with your actual contact information.
5. **Images**: The `images/` directory is optional. If you upload the confusion matrix images (e.g., from your previous input), link them here.
6. **License**: Add a `LICENSE` file to the repository if not already present.

### How to Use
1. **Create the Repository**: If not already done, create a GitHub repository at `https://github.com/SairamNagarajan1/DiseasePrediction`.
2. **Add Files**: Upload the `disease_prediction.py` script, dataset, and optionally the `requirements.txt` and image files.
3. **Paste README**: Copy the above content into a `README.md` file in the repository root.
4. **Commit and Push**: Use Git to commit and push the changes:
   ```bash
   git add .
   git commit -m "Add README and project files"
   git push origin main
