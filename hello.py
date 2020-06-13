import pandas as pd
import numpy as np
import os
import os.path


    #scales features
def featureScaling(x):
        range = np.ptp(x)
        mean = np.mean(x)
        return np.divide(np.subtract(x,mean), range)

    #calculate cost of each iteration
def costFunction(x, y, t):
    difference = np.subtract(np.matmul(x,theta),y)
    partial_derivative = np.square(difference)
    cost = partial_derivative*(1/2*y.shape[0])
    return cost

def gradientDescent(x,y,theta,alpha,iterations):
    for i in range(iterations):
        difference = np.subtract(np.matmul(x,theta),y)
        product = np.matmul(x.transpose(),difference)
        theta = theta -(alpha/y.shape[0])*(product)

    

#create array for output (RDGP/capita)
output_file = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv"
with open(output_file) as f:
        row_count = sum(1 for line in f)-5
df = pd.read_csv(output_file, skiprows=4)
y = np.zeros((row_count,1))
y[:,0] = np.array(df['2017'])

#create array for inputs
inputs = np.zeros((row_count,11))
inputs[:,0] = np.ones(row_count)
print(inputs[1,0])
path = 'C:\\Users\\bmisr\\VSCodeFiles\\python'
folder = os.fsencode(path)

#iterate through all the parameter files 
filenames = []   
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith('.csv') and filename != 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv': 
        filenames.append(filename)

#add data from 2017 to array of inputs
count =1
for name in filenames:
    df = pd.read_csv(name, skiprows=4)
    new_col = np.array(df['2017'])
    inputs[:,count] = new_col
    count+=1
print(count)

#initialize theta
theta = np.zeros((count,1))
print(costFunction(inputs, y, theta))
gradientDescent(inputs, y, theta, 0.01, 1500)
print(theta)






       
    
    
     
    