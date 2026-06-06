# Smart City Health & Environment
# Regression & Classification with Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# 1. Load Dataset in Chunks
# ----------------------------
chunk_size = 100000
chunks = pd.read_csv("AQI Bangladesh.csv", chunksize=chunk_size)

data_list = []
aqi_col_name = None

for chunk in chunks:
    # Clean column names
    chunk.columns = chunk.columns.str.strip().str.replace('\ufeff', '')
    
    # Detect AQI column automatically
    if aqi_col_name is None:
        candidates = [col for col in chunk.columns if 'aqi' in col.lower()]
        if not candidates:
            raise ValueError("AQI column not found in CSV!")
        aqi_col_name = candidates[0]
    
    # Convert AQI to numeric
    chunk[aqi_col_name] = pd.to_numeric(chunk[aqi_col_name], errors='coerce')
    
    # Keep only numeric columns
    numeric_chunk = chunk.select_dtypes(include=[np.number])
    
    data_list.append(numeric_chunk)

# Combine chunks
data = pd.concat(data_list, ignore_index=True)

# ----------------------------
# 2. Drop rows where AQI is NaN
# ----------------------------
data = data.dropna(subset=[aqi_col_name])

# ----------------------------
# 3. Reduce dataset size for performance
# ----------------------------
if len(data) > 300000:
    data = data.sample(n=300000, random_state=42)

# ----------------------------
# 4. Feature Selection
# ----------------------------
target = aqi_col_name
numeric_cols = data.select_dtypes(include=[np.number]).columns
features = numeric_cols.drop(target)

X = data[features]
y_reg = data[target]

# Fill remaining NaNs in features
X = X.fillna(X.mean())

# ----------------------------
# 5. Pipeline for scaling and imputing features
# ----------------------------
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_scaled = pipeline.fit_transform(X)

# ----------------------------
# 6. Train-Test Split (Regression)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# ----------------------------
# 7. Regression Models
# ----------------------------
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred_knn = knn_reg.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

print("=== Regression Results ===")
print("Linear Regression RMSE:", rmse_lr)
print("KNN Regression RMSE:", rmse_knn)
print("Decision Tree RMSE:", rmse_dt)

# ----------------------------
# 8. Plot Regression Results
# ----------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, alpha=0.3, label='Linear Regression', color='blue')
plt.scatter(y_test, y_pred_knn, alpha=0.3, label='KNN Regression', color='green')
plt.scatter(y_test, y_pred_dt, alpha=0.3, label='Decision Tree', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Regression: Predicted vs Actual AQI')
plt.legend()
plt.show()

# ----------------------------
# 9. Health Risk Classes
# ----------------------------
def risk_level(aqi):
    if aqi <= 50:
        return 0   # Low
    elif aqi <= 100:
        return 1   # Medium
    else:
        return 2   # High

data['Health_Risk'] = data[target].apply(risk_level)
y_clf = data['Health_Risk']

# ----------------------------
# 10. Train-Test Split (Classification)
# ----------------------------
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42
)

# ----------------------------
# 11. Classification Models
# ----------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
y_pred_log = log_reg.predict(X_test_clf)

# KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_clf, y_train_clf)
y_pred_knn = knn_clf.predict(X_test_clf)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_clf, y_train_clf)
y_pred_dt = dt_clf.predict(X_test_clf)

print("\n=== Classification Results ===")
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print("KNN Classifier Accuracy:", accuracy_score(y_test_clf, y_pred_knn))
print("Decision Tree Accuracy:", accuracy_score(y_test_clf, y_pred_dt))

print("\nF1 Scores:")
print("Logistic Regression F1:", f1_score(y_test_clf, y_pred_log, average='weighted'))
print("KNN Classifier F1:", f1_score(y_test_clf, y_pred_knn, average='weighted'))
print("Decision Tree F1:", f1_score(y_test_clf, y_pred_dt, average='weighted'))

# ----------------------------
# 12. Classification Confusion Matrices
# ----------------------------
for model_name, y_pred in zip(
    ['Logistic Regression', 'KNN Classifier', 'Decision Tree'], 
    [y_pred_log, y_pred_knn, y_pred_dt]
):
    cm = confusion_matrix(y_test_clf, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low','Medium','High'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# ----------------------------
# 13. Summary Bar Plots
# ----------------------------
# Regression RMSE
reg_models = ['Linear', 'KNN', 'Decision Tree']
rmse_values = [rmse_lr, rmse_knn, rmse_dt]

plt.figure(figsize=(6,4))
plt.bar(reg_models, rmse_values, color=['blue','green','red'])
plt.ylabel('RMSE')
plt.title('Regression Model RMSE Comparison')
plt.show()

# Classification F1
clf_models = ['Logistic', 'KNN', 'Decision Tree']
f1_values = [
    f1_score(y_test_clf, y_pred_log, average='weighted'),
    f1_score(y_test_clf, y_pred_knn, average='weighted'),
    f1_score(y_test_clf, y_pred_dt, average='weighted')
]

plt.figure(figsize=(6,4))
plt.bar(clf_models, f1_values, color=['blue','green','red'])
plt.ylabel('F1 Score')
plt.title('Classification Model F1 Score Comparison')
plt.show()
