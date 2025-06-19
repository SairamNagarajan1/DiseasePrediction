# Import libraries for data processing, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from statistics import mode  # Use statistics.mode for non-numeric data

# Step 1: Load and prepare dataset
# Load CSV file
df = pd.read_csv('/content/improved_disease_dataset.csv')

# Encode disease labels to numbers
label_encoder = LabelEncoder()
df['disease'] = label_encoder.fit_transform(df['disease'])

# Separate features and target
X_data = df.drop(columns=['disease'])
y_data = df['disease']

# Plot initial class distribution
plt.figure(figsize=(15, 5))
sns.countplot(x=y_data)
plt.title('Disease Class Distribution (Before Balancing)')
plt.xticks(rotation=45)
plt.show()

# Balance dataset using oversampling
oversampler = RandomOverSampler(random_state=42)
X_bal, y_bal = oversampler.fit_resample(X_data, y_data)

# Display balanced class counts
print("Balanced Class Distribution:\n", pd.Series(y_bal).value_counts())

# Step 2: Preprocess data
# Encode categorical columns if present
if 'gender' in X_bal.columns:
    X_bal['gender'] = LabelEncoder().fit_transform(X_bal['gender'])

# Fill missing values
X_bal = X_bal.fillna(0)

# Ensure target is 1D
y_bal = np.ravel(y_bal)

# Step 3: Cross-validate models
# Define classifiers
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

# Configure stratified k-fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate models
for model_name, model in models.items():
    try:
        scores = cross_val_score(model, X_bal, y_bal, cv=skfold, scoring='accuracy', n_jobs=-1)
        print("=" * 45)
        print(f"Algorithm: {model_name}")
        print(f"CV Scores: {scores}")
        print(f"Average Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print("=" * 45)
        print(f"Algorithm {model_name} failed: {e}")

# Step 4: Train and evaluate individual models
# Initialize classifiers
svm_model = SVC()
nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=42)

# Train and evaluate SVM
svm_model.fit(X_bal, y_bal)
svm_y_pred = svm_model.predict(X_bal)
svm_cm = confusion_matrix(y_bal, svm_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges')
plt.title('SVM Confusion Matrix')
plt.show()
print(f"SVM Accuracy: {accuracy_score(y_bal, svm_y_pred) * 100:.2f}%")

# Train and evaluate Naive Bayes
nb_model.fit(X_bal, y_bal)
nb_y_pred = nb_model.predict(X_bal)
nb_cm = confusion_matrix(y_bal, nb_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Naive Bayes Confusion Matrix')
plt.show()
print(f"Naive Bayes Accuracy: {accuracy_score(y_bal, nb_y_pred) * 100:.2f}%")

# Train and evaluate Random Forest
rf_model.fit(X_bal, y_bal)
rf_y_pred = rf_model.predict(X_bal)
rf_cm = confusion_matrix(y_bal, rf_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Random Forest Confusion Matrix')
plt.show()
print(f"Random Forest Accuracy: {accuracy_score(y_bal, rf_y_pred) * 100:.2f}%")

# Step 5: Ensemble predictions
# Combine predictions using majority voting
ensemble_y_pred = [mode([s, n, r]) for s, n, r in zip(svm_y_pred, nb_y_pred, rf_y_pred)]
ensemble_cm = confusion_matrix(y_bal, ensemble_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Ensemble Confusion Matrix')
plt.show()
print(f"Ensemble Accuracy: {accuracy_score(y_bal, ensemble_y_pred) * 100:.2f}%")

# Step 6: Symptom-based prediction function
def predict_disease(symptoms_input):
    # Parse symptoms
    symptom_list = [s.strip() for s in symptoms_input.split(',')]
    # Initialize feature vector
    feature_vector = np.zeros(len(X_data.columns))
    feature_map = {name: idx for idx, name in enumerate(X_data.columns)}
    
    # Set present symptoms to 1
    for symptom in symptom_list:
        if symptom in feature_map:
            feature_vector[feature_map[symptom]] = 1
    
    # Reshape for prediction
    feature_vector = feature_vector.reshape(1, -1)
    
    # Create DataFrame to preserve feature names
    input_df = pd.DataFrame(feature_vector, columns=X_data.columns)
    
    # Get predictions
    rf_pred = label_encoder.inverse_transform(rf_model.predict(input_df))[0]
    nb_pred = label_encoder.inverse_transform(nb_model.predict(input_df))[0]
    svm_pred = label_encoder.inverse_transform(svm_model.predict(input_df))[0]
    
    # Combine predictions using majority voting
    final_pred = mode([rf_pred, nb_pred, svm_pred])
    
    return {
        'RandomForest': rf_pred,
        'NaiveBayes': nb_pred,
        'SVM': svm_pred,
        'FinalPrediction': final_pred
    }

# Test prediction
print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))
