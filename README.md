# Disease Prediction Using Machine Learning

This repository contains a machine learning project for predicting diseases based on symptom inputs. The project implements a comprehensive pipeline that includes data preprocessing, class balancing, training of multiple machine learning models (SVM, Naive Bayes, Random Forest), ensemble prediction using majority voting, and visualization of results through confusion matrices. The code is designed to work with the `improved_disease_dataset.csv` dataset and includes a function to predict diseases from user-provided symptoms.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
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

### Dependencies
Install the required Python packages using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
Setup
Clone the repository:
bash

Collapse

Wrap

Run

Copy
git clone https://github.com/SairamNagarajan1/DiseasePrediction.git
cd DiseasePrediction
Ensure the dataset improved_disease_dataset.csv is placed in the /content/ directory (e.g., in Google Colab) or adjust the file path in the script to match your local setup (e.g., pd.read_csv('path/to/your/dataset.csv')).
Run the script with the installed dependencies.

Usage
Run the Script: Execute the main Python script to process the dataset, train models, and generate visualizations:
bash

Collapse

Wrap

Run

Copy
python disease_prediction.py
This will display class distribution plots, confusion matrices, cross-validation scores, and model accuracies.
Predict Diseases: Use the predict_disease function to diagnose based on symptoms. Example:
python

Collapse

Wrap

Run

Copy
from disease_prediction import predict_disease
symptoms = "Itching,Skin Rash,Nodal Skin Eruptions"
result = predict_disease(symptoms)
print(result)
This returns a dictionary with predictions from each model (SVM, Naive Bayes, Random Forest) and the final ensemble diagnosis.
Visualizations: The script automatically generates plots for class distribution and confusion matrices, which are displayed or saved depending on your environment (e.g., Jupyter Notebook, Colab, or local IDE).
Dataset
Source: The project uses improved_disease_dataset.csv, which should contain a disease column (target) and symptom-related features (e.g., "Itching", "Skin Rash").
Format: CSV file with 38 classes (labeled 0-37), balanced to 90 samples per class after oversampling.
Location: Place the file in /content/ or update the path in the script if using a different location.
