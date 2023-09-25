# import dependencies
import numpy as np
import pandas as pd  
from sklearn import datasets
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# loading dataset
boston_dataset = pd.read_csv("./house_price_prediction/boston.csv")
print(boston_dataset.head())

# checking the number of rows and columns
print(boston_dataset.shape)

# checking of there are any null values
print(boston_dataset.isnull().sum())

# statistical measures of the dataset
print(boston_dataset.describe())

# find correlation between each features
correlation = boston_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size':8}, cmap="Blues")   # heatmap is useful in finding the correlation between various features
plt.show()

#splitting the datset into data and target
X = boston_dataset.drop(['MEDV'], axis=1)
Y = boston_dataset['MEDV']
print(X)
print(Y)

# splitting the dataset into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=2)

# training the model
model = XGBRegressor()
model.fit(X_train, Y_train)

'''
note:-
The accuracy score is a metric specifically designed for classification problems, not for regression problems.
In a classification problem, the goal is to predict discrete categories or labels, such as predicting whether
an email is spam or not spam. Accuracy measures the proportion of correctly classified instances out of the
total instances.

In contrast, a regression problem involves predicting continuous numerical values, such as predicting house prices,
stock prices, or temperature. In regression, the target variable is not a category or label; instead, it can take on
a wide range of real-number values. Therefore, the concept of "accuracy" doesn't apply in the same way it does for 
classification.
'''

# evaluation
X_train_prediction = model.predict(X_train)

# R squared error
score_1 = metrics.r2_score(Y_train, X_train_prediction)
print("R squared error: ", score_1)
'''
R2 Close to 1: An R2 value that is very close to 1 suggests that your model is doing an excellent job of explaining the varianc
e in the target variable. It indicates a strong fit, and the model's predictions closely match the actual values.

R2 Close to 0: An R2 value that is close to 0 suggests that your model is not explaining much of the variance in the target
 variable. This could indicate that the model is not a good fit for the data.
'''

# mean absolute error :- it basically finds the difference between original data and actual prediction
score_2 = metrics.mean_absolute_error(Y_train, X_train_prediction)
print("Mean absolute error: ", score_2)
'''
R-squared (R2) Value:
An R2 value very close to 1 (in your case, 0.999995) suggests that your model is explaining almost all of the variance in the target variable. It indicates an extremely strong fit between the model's predictions and the actual values.
An R2 value of 1 would mean a perfect fit, but your value is already very close to perfection. This suggests that your model is doing an excellent job of capturing the patterns in the data.

Mean Absolute Error (MAE):
A MAE of approximately 0.0146 indicates that, on average, your model's predictions are off by only 0.0146 units from the actual values.
This is a very low MAE and indicates that your model's predictions are highly accurate, as the magnitude of the errors is quite small.
'''

# visualizing the actual prices and predicted prices
plt.scatter(Y_train,X_train_prediction)   # Y_train:-actual prices , X_train_prediction:-predicted prices
plt.xlabel("Actual Price")
plt.ylabel("predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()




# evaluation of test data
X_test_prediction = model.predict(X_test)
test_score1 = metrics.r2_score(Y_test, X_test_prediction)
print("R squared error: ", test_score1)

# mean absolute error :- it basically finds the difference between original data and actual prediction
test_score2 = metrics.mean_absolute_error(Y_test, X_test_prediction)
print("Mean absolute error: ", test_score2)

# visualizing the actual prices and predicted prices
plt.scatter(Y_test,X_test_prediction)   # Y_test:-actual prices , X_test_prediction:-predicted prices
plt.xlabel("Actual Price")
plt.ylabel("predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()