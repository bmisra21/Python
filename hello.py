import pandas as pd
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import functions_for_project as fnc

#create array for output (RDGP/capita)
output_file = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv"
with open(output_file) as f:
        row_count = sum(1 for line in f)-5
df = pd.read_csv(output_file, skiprows=4)
y = np.zeros((row_count,1))
y[:,0] = np.array(df['2017'])

#create array for inputs
inputs = np.zeros((row_count,6), dtype = 'float64')
inputs[:,0] = np.ones(row_count)
path = 'C:\\Users\\bmisr\\VSCodeFiles\\python'
folder = os.fsencode(path)

#iterate through all the parameter files 
filenames = ['API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_1121956.csv', 'API_SP.DYN.LE00.IN_DS2_en_csv_v2_1120968.csv', 'API_SH.DYN.MORT_DS2_en_csv_v2_1125887.csv', 'API_SH.STA.MMRT_DS2_en_csv_v2_1124457.csv', 'API_SL.AGR.EMPL.MA.ZS_DS2_en_csv_v2_1129927.csv']   


#add data from 2017 to array of inputs
count =1
for name in filenames:
    df = pd.read_csv(name, skiprows=4)
    new_col = np.array(df['2017'])
    inputs[:,count] = new_col
    count+=1

#clean data
idx = fnc.cleanData(inputs)
black_list = (fnc.checkNull(idx,1,y))
print(len(black_list))
for i in range(2010, 2018):
    inputs = fnc.replaceYear(inputs,idx,str(i),filenames)
    idx = fnc.cleanData(inputs)
    black_list = fnc.checkNull(idx,1,y)
    print(len(black_list))
usable_rows = inputs.shape[0]-len(black_list)
new_inputs = np.ones((usable_rows,inputs.shape[1]))
new_outputs = np.ones((usable_rows,1))
count=0
for i in range(inputs.shape[0]):
    if i in black_list:
        continue
    else:
        new_inputs[count,:] = inputs[i,:]
        new_outputs[count,:] = y[i,:]
        count+=1
inputs = new_inputs
inputs_normalize = np.divide(inputs, np.ptp(inputs))
y = new_outputs
outputs_normalize = np.divide(y, np.ptp(inputs))

#graph data
for col in range(1,inputs.shape[1]):
    fnc.plotData(col,y)

#initialize theta
theta = np.zeros((inputs.shape[1],1))
print(fnc.costFunction(inputs_normalize, outputs_normalize, theta))
theta = fnc.gradientDescent(inputs_normalize, outputs_normalize, theta, 20, 10000)
print(fnc.costFunction(inputs_normalize, outputs_normalize, theta))
x = np.zeros((6,1))
x[:,0] = [1, 100, 98.322, 6.7, 19, 1.993]
print(inputs)
print(fnc.hypothesis(theta, x, inputs))
print(np.amax(inputs)-np.amin(inputs))









       
    
    
     
    