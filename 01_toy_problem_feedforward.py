import numpy as np
import os
#print(os.getcwd()) 

with open('data_toy_problem/data_dark_bright_test_4000.csv') as f:
    data_list = f.readlines()

def sigma(z):
    z /= 255
    return 1 / (1 + np.exp(-z))

wa = np.array([[-0.3, -0.7, -0.9, -0.9], [-1, -0.6, -0.6, -0.6], [0.8, 0.5, 0.7, 0.8]])
wb = np.array([[2.6, 2.1, -1.2], [-2.3, -2.3, 1.1]])

count = 0
for line in data_list:
    x = line.split(",")
    target = int(x[0])
    
    for i in range(len(x)):
        x[i] = int(x[i])
    x = x[1:] 

    # x = np.array(x)
    # x = x.reshape(-1,1)
    
    hidden = sigma(np.dot(wa, x))
        
    output = sigma(np.dot(wb,hidden))

    if np.argmax(output) == target:
        count += 1

percent = 100 / len(data_list)*count
    
print(percent)






