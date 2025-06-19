# Import essential libraries for data analysis, visualization, and modeling
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
from scipy import stats

# Step 1: Load and preprocess dataset
# Read the CSV file
dataset = pd.read_csv('/content/improved_disease_dataset.csv')

# Encode disease labels to numerical values
lbl_encoder = LabelEncoder()
dataset['disease'] = lbl_encoder.fit_transform(dataset['disease'])

# Split into features and target
X_features = dataset.drop('disease', axis=1)
y_target = dataset['disease']

# Visualize class distribution before balancing
plt.figure(figsize=(15, 5))
sns.countplot(x=y_target)
plt.title('Class Distribution of Diseases (Pre-Balancing)')
plt.xticks(rotation=45)
plt.show()

# Balance classes using RandomOverSampler
balancer = RandomOverSampler(random_state=42)
X_balanced, y_balanced = balancer.fit_resample(X_features, y_target)

# Print balanced class counts
print("Balanced Class Distribution:\n", pd.Series(y_balanced).value_counts())

# Step 2: Preprocess data
# Encode categorical columns if present (e.g., gender)
if 'gender' in X_balanced.columns:
    X_balanced['gender'] = LabelEncoder().fit_transform(X_balanced['gender'])

# Handle missing values by filling with 0
X_balanced = X_balanced.fillna(0)

# Ensure target is 1D
y_balanced = np.ravel(y_balanced)

# Step 3: Cross-validate models
# Define classifiers
algorithms = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

# Set up stratified k-fold
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
for algo_name, algo in algorithms.items():
    try:
        cv_results = cross_val_score(algo, X_balanced, y_balanced, cv=k_fold, scoring='accuracy', n_jobs=-1)
        print("=" * 45)
        print(f"Algorithm: {algo_name}")
        print(f"CV Scores: {cv_results}")
        print(f"Average Accuracy: {cv_results.mean():.4f}")
    except Exception as e:
        print("=" * 45)
        print(f"Algorithm {algo_name} failed: {e}")

# Step 4: Train and evaluate individual classifiers
# Initialize models
svm_classifier = SVC()
nb_classifier = GaussianNB()
rf_classifier = RandomForestClassifier(random_state=42)

# Train and evaluate SVM
svm_classifier.fit(X_balanced, y_balanced)
svm_preds = svm_classifier.predict(X_balanced)
svm_conf_matrix = confusion_matrix(y_balanced, svm_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('SVM Confusion Matrix')
plt.show()
print(f"SVM Accuracy: {accuracy_score(y_balanced, svm_preds) * 100:.2f}%")

# Train and evaluate Naive Bayes
nb_classifier.fit(X_balanced, y_balanced)
nb_preds = nb_classifier.predict(X_balanced)
nb_conf_matrix = confusion_matrix(y_balanced, nb_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Naive Bayes Confusion Matrix')
plt.show()
print(f"Naive Bayes Accuracy: {accuracy_score(y_balanced, nb_preds) * 100:.2f}%")

# Train and evaluate Random Forest
rf_classifier.fit(X_balanced, y_balanced)
rf_preds = rf_classifier.predict(X_balanced)
rf_conf_matrix = confusion_matrix(y_balanced, rf_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.show()
print(f"Random Forest Accuracy: {accuracy_score(y_balanced, rf_preds) * 100:.2f}%")

# Step 5: Combine predictions for ensemble
# Use majority voting to combine predictions
ensemble_preds = [stats.mode([s, n, r])[0] for s, n, r in zip(svm_preds, nb_preds, rf_preds)]
ensemble_conf_matrix = confusion_matrix(y_balanced, ensemble_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(ensemble_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Ensemble Model Confusion Matrix')
plt.show()
print(f"Ensemble Accuracy: {accuracy_score(y_balanced, ensemble_preds) * 100:.2f}%")

# Step 6: Define prediction function
def diagnose_symptoms(symptom_string):
    # Parse input symptoms
    symptoms = [s.strip() for s in symptom_string.split(',')]
    # Create input vector
    input_vector = np.zeros(len(X_features.columns))
    symptom_indices = {col: idx for idx, col in enumerate(X_features.columns)}
    
    # Set 1 for present symptoms
    for sym in symptoms:
        if sym in symptom_indices:
            input_vector[symptom_indices[sym]] = 1
    
    # Reshape for prediction
    input_vector = input_vector.reshape(1, -1)
    
    # Get individual predictions
    rf_diag = lbl_encoder.inverse_transform(rf_classifier.predict(input_vector))[0]
    nb_diag = lbl_encoder.inverse_transform(nb_classifier.predict(input_vector))[0]
    svm_diag = lbl_encoder.inverse_transform(svm_classifier.predict(input_vector))[0]
    
    # Combine predictions
    final_diag = stats.mode([rf_diag, nb_diag, svm_diag])[0]
    
    return {
        'RandomForest': rf_diag,
        'NaiveBayes': nb_diag,
        'SVM': svm_diag,
        'FinalDiagnosis': final_diag
    }

# Test prediction
print(diagnose_symptoms("Itching,Skin Rash,Nodal Skin Eruptions"))
