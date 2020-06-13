import csv
import pandas as pd
import numpy as np
import os
import os.path
from os import path

#create array for output (RDGP/capita)
filename = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv"

df = pd.read_csv(filename, skiprows=4)
print (df['2018'][1])
output = np.array(df['2018'])


#create vectors for inputs
row_count = sum(1 for row in "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv")
dimensions = (row_count-5,1)
inputs = np.ones(dimensions)
path = 'C:\\Users\\bmisr\\VSCodeFiles\\python'
folder = os.fsencode(path)

filenames = []   
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.csv') ): 
        filenames.append(filename)
for name in filenames:
    print(name)
    if os.path.isfile(name):
        print ("File exist")
    else:
        print ("File not exist")
    
    
     
    