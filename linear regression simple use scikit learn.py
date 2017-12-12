import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#y = mx + b is linear Regression

points = np.genfromtxt('e:/ml/data.csv', delimiter=',') #read data from file data.csv
X, Y = np.array(points[:,0])[:, np.newaxis], np.array(points[:,1]) # X = column 1 and rotation axis, Y = column 2


model = LinearRegression().fit(X, Y)
m = model.coef_[0]
b = model.intercept_
print("m = ", m, " b = ", b)

x_test = np.linspace(np.min(X), np.max(X))
y_pred = model.predict(x_test[:,None])

plt.scatter(X, Y)
plt.plot(x_test, y_pred, 'r')
plt.legend(['Predicted', 'observed data'])
plt.show()