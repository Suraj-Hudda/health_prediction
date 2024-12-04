import matplotlib.pyplot as plt

# Model names and their corresponding accuracies
model_names = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Naive Bayes"
]

accuracies = [
    58.33,  # Logistic Regression
    70.83,  # K-Nearest Neighbors
    66.67,  # Support Vector Machine
    62.50,  # Decision Tree
    66.67,  # Random Forest
    54.17,  # Gradient Boosting
    66.67   # Naive Bayes
]

# Create a histogram (bar plot)
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Performance', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylim(0, 100)  # Accuracy is typically between 0-100%

# Display the accuracy values on top of each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
