import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset into a Pandas dataframe
df = pd.read_csv("default_of_credit_card_clients.csv", header=1)
df.drop('ID', axis=1, inplace=True) # drop ID column as it does not contribute to the prediction

# Separate the independent variables (features) from the dependent variable (target)
X = df.iloc[:, :-1] # independent variables
y = df.iloc[:, -1] # dependent variable (target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier on the training data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the testing data
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
