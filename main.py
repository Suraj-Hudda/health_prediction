import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # You can choose other classifiers
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('transposed_health_data.csv')

# 2. Split into features and target
X = df.iloc[:, :-1]  # First 19 columns (features)
y = df.iloc[:, -1]   # Last column (target)

# 3. Preprocessing (Optional but recommended)
# If your dataset has missing values, you can impute them or drop rows with missing values
X = X.fillna(X.mean())  # Example: Filling missing values with column mean

# If your features have different scales, you may want to scale them
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train a classification model (e.g., Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model as a .pkl file
joblib.dump(clf, 'classification_model.pkl')

# Optionally, you can also save the scaler if you're using it to preprocess future data
joblib.dump(scaler, 'scaler.pkl')
