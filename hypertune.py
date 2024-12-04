import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('transposed_health_data.csv')

# 2. Split into features (X) and target (y)
X = df.iloc[:, :-1]  # First 19 columns (features)
y = df.iloc[:, -1]   # Last column (target)

# 3. Preprocessing
# Handle missing values by filling with column mean (works for numeric data)
X = X.fillna(X.mean())  # Example: Filling missing values with column mean

# Optional: Scaling features
# RandomForest does not require scaling, but it's good for other models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Define the parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required at leaf nodes
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    'bootstrap': [True, False]  # Bootstrap samples when building trees
}

# 6. Create the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# 7. Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# 8. Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# 9. Get the best parameters and model from GridSearchCV
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

# 10. Evaluate the best model on the test set
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Tuned Model: {accuracy * 100:.2f}%")

# 11. Save the tuned model to a .pkl file
joblib.dump(best_rf_model, 'tuned_classification_model.pkl')
