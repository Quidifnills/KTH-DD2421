"""
Preprocessing & Manual Classification — Annotated Version

This annotated script is generated from the original notebook: preprocessing_classify_manually.ipynb.
It preserves the original code while adding clear, submission-ready comments.

Structure:
- Reproducible imports and configuration
- Data loading and basic sanity checks
- Feature engineering and preprocessing (encoding, scaling, imputation)
- Pipeline construction and cross-validation
- Final training, evaluation, and artifact export

Conventions:
- Follows scikit-learn's fit/transform/predict API.
- Uses stratified splits for classification tasks when labels are imbalanced.
- Reports both Accuracy and F1-score (macro/weighted) where applicable.

How to run:
    python preprocessing_classify_manually_annotated.py

Notes:
- If your code expects relative paths, run from the project root or adjust paths below.
- All third-party libraries must be installed in the active environment.
"""


# ----- Notebook Markdown -----
# # DD2421 Machine Learning: Programming Challenge

# === Cell 1: Imports and environment setup ===
# Set up dependencies; keep imports at top for clarity and performance.
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import time


# === Cell 2: Load dataset(s) from disk ===
# Read raw data into memory; prefer explicit dtypes and parse dates if needed.
df = pd.read_csv('TrainOnMe_orig.csv')
df.head()


# === Cell 3: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.info()


# === Cell 4: General processing / utilities ===
# Helper utilities or miscellaneous processing.
# Handling NaN
df.isna().any().any()   # True 代表表里存在至少一个 NaN/NaT


# === Cell 5: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.shape


# === Cell 6: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.drop_duplicates().shape


# === Cell 7: General processing / utilities ===
# Helper utilities or miscellaneous processing.

df.isnull().sum()


# === Cell 8: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.describe(include="all")


# === Cell 9: Visualization / plotting of results ===
# Plot distributions, correlations, and performance curves to diagnose behavior.
# Select only the original numerical columns for outlier detection
numerical_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x8', 'x9', 'x10', 'x11', 'x13']

fig, axs = plt.subplots(len(numerical_cols), 1, figsize=(11, 18), dpi=95)
for i, col in enumerate(numerical_cols):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
plt.tight_layout()
plt.show()


# === Cell 10: General processing / utilities ===
# Helper utilities or miscellaneous processing.
# # List of numerical columns to check for outliers
# numerical_cols = ['x1', 'x2', 'x3', 'x4', 'x6', 'x8', 'x9', 'x10', 'x11']
# # numerical_cols = ['x8', 'x9']

# print(f"Original shape of the dataframe: {df.shape}")

# # Loop through each numerical column to find and remove outliers
# for col in numerical_cols:
#     # Calculate IQR
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1

#     # Define outlier boundaries
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Identify outliers
#     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
#     # Remove the outlier rows
#     df = df.drop(outliers.index)
# print(f"Shape of the dataframe after removing the outliers: {df.shape}")


# === Cell 11: Visualization / plotting of results ===
# Plot distributions, correlations, and performance curves to diagnose behavior.
# Select only the original numerical columns for outlier detection
# numerical_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x8', 'x9', 'x10', 'x11', 'x13']

# fig, axs = plt.subplots(len(numerical_cols), 1, figsize=(11, 18), dpi=95)
# for i, col in enumerate(numerical_cols):
#     axs[i].boxplot(df[col], vert=False)
#     axs[i].set_ylabel(col)
# plt.tight_layout()
# plt.show()


# === Cell 12: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df['y'].value_counts()


# === Cell 13: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df['x7'].value_counts()


# === Cell 14: Visualization / plotting of results ===
# Plot distributions, correlations, and performance curves to diagnose behavior.
print("Each category within the y column has the following pie chart:")
print(df.groupby(['y']).size())

plt.pie(df['y'].value_counts(), labels=[
        'Andjorg', 'Andsuto', 'Jorgsuto'], autopct='%.f%%', shadow=True)
plt.title('Label Proportionality')
plt.show()


# ----- Notebook Markdown -----
# ## Preprocess categorical input variables

# === Cell 15: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df['x12'].value_counts()
unique_value_count = df['x12'].nunique()
print(f"The 'x12' column has {unique_value_count} unique value(s).")
if unique_value_count == 1:
    # axis=1 specifies we are dropping a column
    # inplace=True modifies the DataFrame directly
    df.drop('x12', axis=1, inplace=True)
    print("Column 'x12' has been successfully removed.")


# === Cell 16: Encode categorical features ===
# Convert categorical variables to numeric representation for ML algorithms.
# One hot coding for x7
# drop_first=True: To Avoid Multicollinearity (For Linear Models)
df_dummy = pd.get_dummies(df['x7'], prefix='x7',drop_first=True)
df = pd.concat([df, df_dummy], axis=1).drop(['x7'], axis=1)


# === Cell 17: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.info()


# ----- Notebook Markdown -----
# ## Correlation Analysis

# === Cell 18: Visualization / plotting of results ===
# Plot distributions, correlations, and performance curves to diagnose behavior.
correlation = df[['x1','x2','x3','x4','x5','x6','x8','x9','x10','x11','x13',
                  'x7_Polkagris','x7_Polskorgris', 'x7_Schottisgris','x7_Slängpolskorgris']].corr()
plt.figure(figsize=(15,10))
sb.heatmap(correlation, cmap='coolwarm', annot = True)


# === Cell 19: General processing / utilities ===
# Helper utilities or miscellaneous processing.
# Drop redundant variables
df.drop(['x5', 'x13'], axis=1, inplace=True)


# === Cell 20: General processing / utilities ===
# Helper utilities or miscellaneous processing.
df.describe()


# === Cell 21: Visualization / plotting of results ===
# Plot distributions, correlations, and performance curves to diagnose behavior.
correlation = df[['x1','x2','x3','x4','x6','x8','x9','x10','x11',
                  'x7_Polkagris','x7_Polskorgris', 'x7_Schottisgris','x7_Slängpolskorgris']].corr()
plt.figure(figsize=(15,10))
sb.heatmap(correlation, cmap='coolwarm', annot = True)


# === Cell 22: Split data into train/validation (or train/test) sets ===
# Ensure fair evaluation; use stratification for classification to preserve label ratios.
X = df.drop(['y'], axis=1)
y= df['y']

# cross validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 分出验证集
RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# === Cell 23: Scale/normalize numerical features ===
# Normalize feature ranges to help distance-based or gradient-based models.
# 2. 定义所有分类器
classifiers = {
    "K-neighbours": KNeighborsClassifier(),
    "Decision tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Extremely random forest": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "Adaboost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Bagging": BaggingClassifier(random_state=RANDOM_STATE),
    "MLP": MLPClassifier(max_iter=2000, hidden_layer_sizes=(20,20), random_state=RANDOM_STATE),
    "SVM (rbf)": SVC(random_state=RANDOM_STATE),
    "SVM (linear)": SVC(kernel="linear", random_state=RANDOM_STATE),
    "SVM (polynomial)": SVC(kernel="poly", random_state=RANDOM_STATE),
    "Ridge Classifier": RidgeClassifierCV()
}

# 3. 存储结果
results = []

for clf_name, clf in classifiers.items():
    print(f"Training: {clf_name}...")
    
    # 交叉验证
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, n_jobs=-1)
    
    # 训练并评估
    clf.fit(X_train_scaled, y_train)
    y_val_pred = clf.predict(X_val_scaled)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    results.append({
        'Classifier': clf_name,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Val_Accuracy': val_acc,
        'Val_F1': val_f1
    })

results_df = pd.DataFrame(results).sort_values('Val_Accuracy', ascending=False)
print("\n" + "="*70)
print(results_df.to_string(index=False))

# ============================================
# 4. 用最佳模型对测试集进行预测
# ============================================
best_model_name = results_df.iloc[0]['Classifier']
best_clf = classifiers[best_model_name]

print(f"\n Best Model: {best_model_name}")

# 用全部训练数据重新训练
scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X)
best_clf.fit(X_all_scaled, y)



# === Cell 24: Load dataset(s) from disk ===
# Read raw data into memory; prefer explicit dtypes and parse dates if needed.
test_df = pd.read_csv('EvaluateOnMe.csv')
test_df.head()


# === Cell 25: General processing / utilities ===
# Helper utilities or miscellaneous processing.
unique_value_count = test_df['x12'].nunique()
print(f"The 'x12' column has {unique_value_count} unique value(s).")
if unique_value_count == 1:
    # axis=1 specifies we are dropping a column
    # inplace=True modifies the DataFrame directly
    test_df.drop('x12', axis=1, inplace=True)
    print("Column 'x12' has been successfully removed.")


# === Cell 26: Encode categorical features ===
# Convert categorical variables to numeric representation for ML algorithms.
# One hot coding for x7
# drop_first=True: To Avoid Multicollinearity (For Linear Models)
test_df_dummy = pd.get_dummies(test_df['x7'], prefix='x7',drop_first=True)
test_df = pd.concat([test_df, test_df_dummy], axis=1).drop(['x7'], axis=1)


# === Cell 27: General processing / utilities ===
# Helper utilities or miscellaneous processing.
# Drop redundant variables
test_df_clean = test_df.drop(['x5', 'x13'], axis=1)


# === Cell 28: General processing / utilities ===
# Helper utilities or miscellaneous processing.
test_df_clean.info()


# === Cell 29: Inference / predictions on validation or test data ===
# Generate predictions for downstream evaluation or submission.
# 预测测试集
X_test_scaled = scaler_final.transform(test_df_clean)
y_test_pred = best_clf.predict(X_test_scaled)

# 保存预测结果
np.savetxt("y_pred.txt", y_test_pred, fmt='%s')
