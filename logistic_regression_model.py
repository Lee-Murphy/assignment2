import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/test_dataset.csv')

# Check number of columns before imputing missing values
print(f"Number of columns before imputing missing values: {data.shape[1]}")

# Check for columns with all missing values
all_missing = data.columns[data.isnull().sum() == data.shape[0]]
if len(all_missing) > 0:
    print("The following columns have all missing values:")
    print(list(all_missing))
    data.drop(columns=all_missing, inplace=True)

# Impute missing values with the most frequent value
imp = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

# Convert categorical variables to numeric
data = pd.get_dummies(data)

# Split the dataset into training and testing sets using hold-out
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target_5clique', axis=1), data['target_5clique'], test_size=0.2, random_state=42)

# Normalize the features using standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a logistic regression classifier
clf = LogisticRegression()

# Define the hyperparameter grid to search over
param_grid = {'C': [0.1, 1, 10, 100]}

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Fit the classifier to the training data using the best hyperparameters
clf_best = grid_search.best_estimator_
clf_best.fit(X_train_scaled, y_train)

# Evaluate the classifier on the testing data using hold-out
accuracy_holdout = clf_best.score(X_test_scaled, y_test)
print(f"Hold-out accuracy: {accuracy_holdout:.4f}")
