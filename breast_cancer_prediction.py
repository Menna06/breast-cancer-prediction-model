# Preprocessing
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv('breast-cancer.csv')

# Encoding categorical data
labelEncoder = LabelEncoder()
labelEncoder.fit(data["diagnosis"])
data["diagnosis"] = labelEncoder.transform(data["diagnosis"])

# Remove duplicates
data = data.drop_duplicates()

# Detecting outliers (Z-score method)
from scipy.stats import zscore
data = data[(zscore(data.select_dtypes(include='number')) < 3).all(axis=1)]

# Define features (X) and target (y)
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
     'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store evaluation metrics
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# Display results
results_df = pd.DataFrame(results).T
results_df.to_csv("results/metrics.csv")
print("Model Performance Metrics:")
print(results_df)

# Model Performance Comparison Plot

plt.figure(figsize=(10, 6))

results_df['Accuracy'].sort_values().plot(
    kind='barh'
)

plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracy Comparison')

plt.tight_layout()

# Save image
plt.savefig("images/model_accuracy_comparison.png")

# Show to user
plt.show()

# Close figure
plt.close()

# Plot confusion matrix for each model
for name, model in models.items():

    if name != "SVM":
        continue

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labelEncoder.classes_
    )

    disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {name}')
plt.tight_layout()
plt.savefig("images/svm_confusion_matrix.png")
plt.close()

# Clear all existing figures to prevent duplication
plt.close('all')

# Feature importance for Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importance:")
print(feature_importance_rf)
print("PASSED RANDOM FOREST")

# Feature importance for XGBoost
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
feature_importance_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("XGBoost Feature Importance:")
print(feature_importance_xgb)

# # Coefficients for Logistic Regression
# print("STARTING LOGISTIC REGRESSION")
# lr_model = LogisticRegression(max_iter=1000, random_state=42)
# print("BEFORE LR FIT")

# lr_model.fit(X_train_scaled, y_train)

# print("AFTER LR FIT")
# coef_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': lr_model.coef_[0]
# }).sort_values(by='Coefficient', ascending=False)
# print("Logistic Regression Coefficients:")
# print(coef_df)

# Coefficients for SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': svm_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("SVM Coefficients:")
print(svm_coef_df)

# Feature importance for KNN using permutation importance
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
perm_importance = permutation_importance(knn_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
feature_importance_knn = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)
print("KNN Feature Importance (Permutation):")
print(feature_importance_knn)

# Random Forest Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.savefig("images/random_forest_importance.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_xgb['Feature'], feature_importance_xgb['Importance'])
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("images/xgboost_importance.png")
plt.close()

# # Logistic Regression Coefficients Plot
# plt.figure(figsize=(10, 6))
# plt.barh(coef_df['Feature'], coef_df['Coefficient'])
# plt.xlabel('Coefficient Value')
# plt.title('Logistic Regression Coefficients')
# plt.gca().invert_yaxis()
# plt.show()

# SVM Coefficients Plot
plt.figure(figsize=(10, 6))
plt.barh(svm_coef_df['Feature'], svm_coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('SVM Coefficients')
plt.gca().invert_yaxis()
plt.savefig("images/svm_coefficients.png")
plt.close()

# KNN Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_knn['Feature'], feature_importance_knn['Importance'])
plt.xlabel('Importance (Mean Accuracy Decrease)')
plt.title('KNN Feature Importance (Permutation)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("images/knn_importance.png")
plt.close()


# Decision Tree Feature Importance Plot
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
feature_importance_dt = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Decision Tree Feature Importance:")
print(feature_importance_dt)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_dt['Feature'], feature_importance_dt['Importance'])
plt.xlabel('Importance')
plt.title('Decision Tree Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("images/decision_tree_importance.png")
plt.close()


# Feature importance for Gaussian Naive Bayes using permutation importance
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
perm_importance_gnb = permutation_importance(gnb_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
feature_importance_gnb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance_gnb.importances_mean
}).sort_values(by='Importance', ascending=False)
print("Gaussian Naive Bayes Feature Importance (Permutation):")
print(feature_importance_gnb)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_gnb['Feature'], feature_importance_gnb['Importance'], color='darkblue')
plt.xlabel('Permutation Importance')
plt.title('Gaussian Naive Bayes Feature Importance (Permutation)')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.tight_layout()
plt.savefig("images/gaussian_nb_importance.png")
plt.close()