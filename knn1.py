import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset

data = pd.read_csv('/home/student/Desktop/iris.csv')

# Display the first few rows of the dataset
print(data.head())

# Features and target variable
x = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

print(x.head())
print(y.head())

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize and train the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

# Predict on the test set
y_pred = classifier.predict(x_test)
print(y_pred)
# Evaluate the classifier
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy Score:")
print(ac)
