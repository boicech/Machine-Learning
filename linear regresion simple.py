import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#author boicech;
#y = b + mx is linear regression simple;

#data = pd.read_csv('e:/ml/ex1data1.txt')
#points = data.values

points = np.genfromtxt('e:/ml/data.csv', delimiter=',')

gd = [0, 0] # gd[0] is initialization value of b; gd[1] is initialization value of m
anpha = 0.0001 # chose anpha = 0.0001 or 0.0003 or 0.001 or 0.003...
loop = 1000 #number of loop

def gradient_Descent(gd, points, anpha, loop):
    temp = [0, 0] #variable temporary of gd
    n = len(points)
    i = 0
    while(i < loop):
        #print(gd)
        temp[0] = gd[0] - anpha/n*sum1(gd, points, n)
        temp[1] = gd[1] - anpha/n*sum2(gd, points, n)
        gd = temp
        i += 1
    return gd

def sum1(gd, points, n):
    sum = 0
    for i in range(n):
        sum += (gd[0] + gd[1]*points[i, 0] - points[i, 1])
    return sum

def sum2(gd, points, n):
    sum = 0
    for i in range(n):
        sum += ((gd[0] + gd[1]*points[i, 0] - points[i, 1])*points[i, 0])
    return sum

theta = gradient_Descent(gd, points, anpha, loop) # b and m
x = np.array(points[:,0]) # get column 0 from points variable
y = np.array(points[:,1]) # get column 1 from points variable

x_pop = np.linspace(x.min(), x.max(), 100) #split x axis to 100 parts
t = theta[0] + (theta[1] * x_pop)
plt.plot(x_pop, t, 'r', label='Prediction')
plt.scatter(x, y, label='Training data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (Kg)')
plt.show()