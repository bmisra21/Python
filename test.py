import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.zeros((3,2))
c = np.ones((2,3))
#print(np.multiply(a,2))
alpha = 0.01
print(alpha/(a.shape[0]))
print (np.subtract(a,c))
print(np.mean(a))
x = np.zeros((11,1))
x[:,0] = [1, 98.71320343, 79.9346, 67.5, 638, 19.81767613, 11.31299973, -421769, 53.2079642, 74.98, 8852859]
print(x)

#count =0
#for file in os.listdir(folder):
    #if (count < 5):
        #filename = os.fsdecode(file)
        #if filename.endswith('.csv') and filename != 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1120951.csv': 
           # filenames.append(filename)
   # count+=1