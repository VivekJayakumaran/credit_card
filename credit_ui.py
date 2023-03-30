# import tkinter as tk
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# # Load the dataset into a Pandas dataframe
# df = pd.read_csv("UCI_Credit_Card.csv", header=1)
# # df.drop('ID', axis=1, inplace=True) # drop ID column as it does not contribute to the prediction

# # Train a Random Forest classifier on the entire dataset
# X = df.iloc[:, :-1] # independent variables
# y = df.iloc[:, -1] # dependent variable (target)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)

# # Create the GUI window
# window = tk.Tk()
# window.title("Credit Card Default Prediction")

# # Add the input fields for the user to enter the feature values
# limit_label = tk.Label(window, text="Credit Limit:")
# limit_label.pack()
# limit_entry = tk.Entry(window)
# limit_entry.pack()

# age_label = tk.Label(window, text="Age:")
# age_label.pack()
# age_entry = tk.Entry(window)
# age_entry.pack()

# # Add a button to trigger the prediction
# def predict():
#     # Get the feature values entered by the user
#     limit = float(limit_entry.get())
#     age = int(age_entry.get())
    
#     # Create a new dataframe with the user input
#     new_data = pd.DataFrame({'LIMIT_BAL': [limit], 'AGE': [age]})
#     new_data.columns = feature_names;
    
#     # Use the trained model to predict the default probability
#     default_prob = rf.predict_proba(new_data)[:, 1][0]
    
#     # Display the prediction result
#     result_label.config(text=f"Default Probability: {default_prob:.2f}")

# predict_button = tk.Button(window, text="Predict", command=predict)
# predict_button.pack()

# # Add a label to display the prediction result
# result_label = tk.Label(window, text="")
# result_label.pack()

# # Run the GUI window
# window.mainloop()
import tkinter as tk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset into a Pandas dataframe
df = pd.read_csv("UCI_Credit_Card.csv", header=1)
# df.drop('ID', axis=1, inplace=True) # drop ID column as it does not contribute to the prediction

# Train a Random Forest classifier on the entire dataset
X = df.iloc[:, :-1] # independent variables
y = df.iloc[:, -1] # dependent variable (target)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Define the feature names used during training
feature_names = X.columns.tolist()

# Create the GUI window
window = tk.Tk()
window.title("Credit Card Default Prediction")

# Add the input fields for the user to enter the feature values
limit_label = tk.Label(window, text="Credit Limit:")
limit_label.pack()
limit_entry = tk.Entry(window)
limit_entry.pack()

age_label = tk.Label(window, text="Age:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

# Add a button to trigger the prediction
def predict():
    # Get the feature values entered by the user
    limit = float(limit_entry.get())
    age = int(age_entry.get())
    
    # Create a new dataframe with the user input
    new_data = pd.DataFrame({'LIMIT_BAL': [limit], 'AGE': [age]})
    
    # Update the column names to match the feature names used during training
    new_data.columns = feature_names
    
    # Use the trained model to predict the default probability
    default_prob = rf.predict_proba(new_data)[:, 1][0]
    
    # Display the prediction result
    result_label.configure(text=f"Default Probability: {default_prob:.2f}")

predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.pack()

# Add a label to display the prediction result
result_label = tk.Label(window, text="")
result_label.pack()

# Run the GUI window
window.mainloop()
