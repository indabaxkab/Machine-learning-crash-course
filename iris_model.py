from sklearn.datasets import load_iris
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import mglearn

iris_dataset = load_iris() #loading the dataset in our program

#The iris object that is returned by load_iris is a Bunch object, which is very similar
#to a dictionary. It contains keys and values:
print(iris_dataset.keys())

#information about our dataset
#print(iris_dataset['DESCR'][:193])

#feature names
#print(iris_dataset['feature_names'])

#lets call the train_split_function and assign the outputs
# X_train contains 75% of the rows of the dataset,
#and X_test contains the remaining 25%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#checking the shape of our X-train ad=nd y_train data
print(f"X_train {X_train.shape}")
print(f"y_train {y_train.shape}")

#checking the shape of the test data for both x and y
print(f"X_test {X_test.shape}")
print(f"y_test {y_test.shape}")

#looking at the data 
#create a dataframe from the data in the X-train
#label the columns using the strings in the iris_dataset.feature name 
iris_dataset = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#create a scatter matrix from the datframe, color by y-train
grr = scatter_matrix(iris_dataset, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()

