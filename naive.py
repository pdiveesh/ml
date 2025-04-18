# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data from CSV
data = pd.read_csv('C:/Users/ADMIN/Downloads/tennisdata.csv')
print("The first 5 rows of the dataset:\n", data.head())

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\nThe first 5 rows of features:\n", X.head())
print("\nThe first 5 values of target:\n", y.head())

# Label Encoding for categorical variables
le = LabelEncoder()
X['Outlook'] = le.fit_transform(X['Outlook'])
X['Temperature'] = le.fit_transform(X['Temperature'])
X['Humidity'] = le.fit_transform(X['Humidity'])
X['Windy'] = le.fit_transform(X['Windy'])
y = le.fit_transform(y)

print("\nEncoded feature set:\n", X.head())
print("\nEncoded target values:\n", y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluate the model
predictions = classifier.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
