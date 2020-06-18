import pandas as pd
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt

#scales features
def featureScaling(x):
        range = np.ptp(x)
        mean = np.mean(x)
        return np.divide(np.subtract(x,mean), range)

#calculate cost of each iteration
def costFunction(x, y, t):
    difference = np.subtract(np.matmul(x,t),y)
    partial_derivative = np.matmul(np.transpose(difference), difference)
    cost = partial_derivative/(2*y.shape[0])
    return cost
    
#gradually moves theta to optimal values
def gradientDescent(x,y,theta,alpha,iterations):
    for i in range(iterations):
        h = np.matmul(x,theta)
        difference = np.subtract(h,y)
        transpose = np.transpose(x)
        product = np.matmul(transpose,difference)
        partial_derivative = np.multiply(product, (alpha/(y.shape[0])))
        theta = np.subtract(theta , partial_derivative)
    return theta

#hypothesis based on theta
def hypothesis(theta, x, inputs):
    x = x*(1/np.ptp(inputs))
    return(np.ptp(inputs)*np.matmul(np.transpose(theta),x))

#return array of zeros and ones based on whether values are present in the array
def cleanData(x):
    idx = np.zeros((x.shape[0],x.shape[1]))
    counter=0
    for row in x:
        id = pd.Index(row)
        idx[counter,:] = id.notnull()
        counter+=1
    return idx
    
#return list that specifies which rows in the dataset have more than a specified number of null values
def checkNull(idx, max_null, y):
    black_list = []
    idy = cleanData(y)
    row_num=0
    for row in idx:
        num_null = 0
        for cell in row:
            if cell == 0:
                num_null+=1
        if num_null >= max_null: 
            black_list.append(row_num)
        row_num+=1
    row_num=0
    for row in idy:
        for cell in row:
            if cell == 0: 
                if (row_num not in black_list):
                    black_list.append(row_num)
        row_num+=1
    return black_list

#If a cell in inputs has a null value, it is replaced with the value of the specified year
def replaceYear(x, idx, year, names):
    for i in range(0,x.shape[0]):
        for j in range(1,x.shape[1]):
            if idx[i,j] == 0:
                df = pd.read_csv(names[j-1], skiprows=4)
                x[i,j] = df[year][i]
    return x

#generate scatter plot for data
def plotData(x,y):
    plt.scatter(x, y[:,0], label="stars", color ="green", marker="1", s=30)
    plt.xlabel('x-axis')
    plt.ylabel('RGDP/Capita')
    plt.title('Graph')
    plt.show()