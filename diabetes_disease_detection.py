import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('diabetes.csv')

print("Dataset Overview:")
data.head()

print("\nSummary Statistics:")
print(data.describe())

print("\nClass Distribution:")
print(data['Outcome'].value_counts())

sns.pairplot(data, hue='Outcome', diag_kind='kde')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='YlOrRd')
plt.title("Correlation Matrix Heatmap")
plt.show()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[feature], kde=True, bins=20)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Outcome', y=feature, data=data)
    plt.title(f"{feature} vs. Outcome")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print("Preprocessed training data:")
print(X_train.head())

print("\nPreprocessed testing data:")
print(X_test.head())

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(random_state=42)

xgb_model.fit(X_train, y_train)
importances = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importances from XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

X_train['Glucose_BMI_interaction'] = X_train['Glucose'] * X_train['BMI']
X_test['Glucose_BMI_interaction'] = X_test['Glucose'] * X_test['BMI']

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with the engineered feature:", accuracy)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

search_method = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
search_method_name = "Random Search"

search_method.fit(X_train, y_train)

print("Best Hyperparameters from", search_method_name, ":", search_method.best_params_)

best_model = search_method.best_estimator_
cv_results = cross_val_score(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print("Cross-Validation Results:", cv_results)
print("Mean CV Accuracy:", np.mean(cv_results))

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Testing Set:", accuracy)

import tensorflow as tf

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, verbose=1)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to class labels (0 or 1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Testing Set:", accuracy)

print("Accuracy - XGBoost:", accuracy)
print("Accuracy - Neural Network:", accuracy)

import shap

explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_test)

explainer_nn = shap.DeepExplainer(model, X_train)
shap_values_nn = explainer_nn.shap_values(X_test)

shap.summary_plot(shap_values_xgb, X_test, plot_type='bar', show=False)
plt.title("XGBoost Feature Importances")
plt.show()

shap.summary_plot(shap_values_nn, X_test, show=False)
plt.title("SHAP Values for Neural Network Predictions")
plt.show()

print("\nComparison:")
print("XGBoost Accuracy:", accuracy)
print("Neural Network Accuracy:", accuracy)
print("XGBoost Interpretability: High (Feature Importances)")
print("Neural Network Interpretability: Medium (SHAP Values)")
print("XGBoost Computational Resources: Low")
print("Neural Network Computational Resources: High")

import joblib

joblib.dump(xgb_model, 'xgboost_diabetes_model.joblib')

model.save('neural_network_diabetes_model.h5')


xgb_loaded_model = joblib.load('xgboost_diabetes_model.joblib')
nn_loaded_model = tf.keras.models.load_model('neural_network_diabetes_model.h5')

sample_data = np.array([[6, 148, 72, 35, 0, 0, 33.6, 0.627, 50]])

xgb_predictions = xgb_loaded_model.predict(sample_data)

nn_predictions = nn_loaded_model.predict(sample_data)

explainer_nn = shap.DeepExplainer(nn_loaded_model, X_train)
shap_values_nn = explainer_nn.shap_values(sample_data)

xgb_binary_prediction = 1 if xgb_predictions > 0.5 else 0
nn_binary_prediction = 1 if nn_predictions > 0.5 else 0

print("XGBoost Prediction:", xgb_binary_prediction)
print("Neural Network Prediction:", nn_binary_prediction)
print("SHAP Values for Neural Network Prediction:", shap_values_nn)