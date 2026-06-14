import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the heart disease dataset
heart_disease = pd.read_csv('heart.csv')


# ------------------------*** STEP 1: PREPROCESSING ***---------------------------------

# Encode categorical text features into numeric values using label encoding indices
mapping_rules = {
    'Sex': {'M': 1, 'F': 0},
    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
    'ExerciseAngina': {'Y': 1, 'N': 0},
    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
}

# Apply explicit engineering dictionary mappings across text features
for col in mapping_rules:
    heart_disease[col] = heart_disease[col].map(mapping_rules[col])
# Handle invalid zero values in clinical indicators by converting them to NaN for imputation
heart_disease['Cholesterol'].replace(0, np.nan, inplace=True)

# Impute missing values in Cholesterol using a 3-Nearest Neighbors strategy
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(heart_disease)
heart_disease = pd.DataFrame(after_impute, columns=heart_disease.columns)

# Handle invalid zero values in Resting Blood Pressure indicator
heart_disease['RestingBP'].replace(0, np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(heart_disease)
heart_disease = pd.DataFrame(after_impute, columns=heart_disease.columns)

# Downcast clinical properties to int32 structural configurations, excluding the Oldpeak float feature
without_Oldpeak = heart_disease.columns
without_Oldpeak = without_Oldpeak.drop('Oldpeak')
heart_disease[without_Oldpeak] = heart_disease[without_Oldpeak].astype('int32')


# --------------------------*** STEP 2: DATA VISUALIZATION ***------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Line Plot: Resting Blood Pressure Trend by Age
plt.figure(figsize=(10,5))
sns.lineplot(data=heart_disease, x='Age', y='RestingBP', color='blue', marker='o')
plt.title('Resting Blood Pressure Trend by Age')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.grid(True)
plt.show()

# 2. Histogram: Distribution of Patient Age
plt.figure(figsize=(10,5))
sns.histplot(data=heart_disease, x='Age', bins=20, color='purple', kde=True)
plt.title('Distribution of Patient Age')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()

# 3. Histogram: Maximum Heart Rate Distribution by Disease Status
plt.figure(figsize=(10,5))
sns.histplot(data=heart_disease, x='MaxHR', hue='HeartDisease', multiple='stack', palette='muted', kde=True)
plt.title('Maximum Heart Rate Distribution by Disease Status')
plt.xlabel('Maximum Heart Rate (MaxHR)')
plt.ylabel('Number of Patients')
plt.show()

# 4. Histogram: Chest Pain Type Broken Down by Heart Disease Status
plt.figure(figsize=(10,5))
temp_df = heart_disease.copy()
temp_df['HeartDisease'] = temp_df['HeartDisease'].map({0: 'No', 1: 'Yes'})

sns.histplot(data=temp_df, x='ChestPainType', hue='HeartDisease', multiple='dodge', palette='Set2')
plt.title('Chest Pain Type Broken Down by Heart Disease Status')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.show()


# -------------------------*** STEP 3: MODEL TRAINING ***------------------------------------
from sklearn.model_selection import train_test_split

# Partition the dataset into training and validation subsets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    heart_disease.drop('HeartDisease', axis=1),
    heart_disease['HeartDisease'],
    test_size=0.2, random_state=42,
    stratify=heart_disease['HeartDisease']
)

# --- Logistic Regression Model Optimization ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
best_solver = ''
test_score = np.zeros(6)

# Evaluate different optimizer solvers to isolate maximum test validation accuracy
for i, n in enumerate(solver):
    lr = LogisticRegression(solver=n).fit(X_train, y_train)
    test_score[i] = lr.score(X_test, y_test)
    if lr.score(X_test, y_test) == test_score.max():
        best_solver = n

# Retrain the final Logistic Regression pipeline utilizing the optimal identified solver
lr = LogisticRegression(solver=best_solver)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f'LogisticRegression Score: {accuracy_score(y_test, lr_pred)}')

# --- Support Vector Machine (SVM) Classification ---
from sklearn.svm import SVC
from sklearn.metrics import f1_score

kernels = {'linear': 0, 'poly': 0, 'rbf': 0, 'sigmoid': 0}
best = '' 

# Optimize structural kernel mapping by verifying performance via weighted F1-score optimization
for i in kernels:
    svm = SVC(kernel=i)
    svm.fit(X_train, y_train)
    yhat = svm.predict(X_test)
    kernels[i] = f1_score(y_test, yhat, average="weighted")
    if kernels[i] == max(kernels.values()):
        best = i

# Retrain the operational Support Vector Classifier with top target kernel configuration
svm = SVC(kernel=best)
svm.fit(X_train, y_train)
svm_predict = svm.predict(X_test)
print(f'SVM f1_score kernel({best}): {f1_score(y_test, svm_predict, average="weighted")}')

# --- Decision Tree Tuning Via Grid Search Strategy ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dtree = DecisionTreeClassifier(class_weight='balanced')
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}

# Execute a 5-fold cross-validation grid search strategy across the hyperparameter surface
grid_search = GridSearchCV(dtree, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Construct final Decision Tree architecture instantiated with the optimized parameters
Ctree = DecisionTreeClassifier(**grid_search.best_params_, class_weight='balanced')
Ctree.fit(X_train, y_train)
dtc_prediction = Ctree.predict(X_test)
print("DecisionTree's Accuracy: ", accuracy_score(y_test, dtc_prediction))

# --- Model Serialization ---
import pickle

# Serialize trained architectures locally as artifact binaries for deployment application usage
with open('logisticR.pkl', 'wb') as file:
    pickle.dump(lr, file)

with open('SVM.pkl', 'wb') as file:
    pickle.dump(svm, file)

with open('DecisionT.pkl', 'wb') as file:
    pickle.dump(Ctree, file)