# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# importing or loading the dataset
dataset = pd.read_csv('..//Check//features.csv')

# distributing the dataset into two components X and Y
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values
# Splitting the X and Y into the
# Training set and Testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_test)

