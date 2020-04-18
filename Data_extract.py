import numpy as np
import math

# space 1, 3, 5, 7, 9, 11, 13, 15, 17

f=open("Dataset\clean.data", "r")
row = 0
for i in f:
    row = row + 1
#print(row)

data = np.zeros((row,10))
temp = []
types = np.zeros(row)
l = 0
f=open("Dataset\clean.data", "r")
#f=open("test.txt", "r")
for i in f:
    a = i.split(' ')
    k = 0
    for j in range(2,17,2):        
        data[l,k] = float(a[j])
        
        #print(l,k)
        k = k + 1
    temp.append(a[18])
    l = l + 1
#print(temp)
f1 = open("Data1.txt","w+")
for i in range(0,row):
    if temp[i] == 'CYT\n':
        data[i,9] = 1
        for j in range(0,10):
            if j < 9:
                f1.write("%f," % (data[i,j]))
            else:
                f1.write("%f\n" % (data[i,j]))
f1.close()

f2 = open("Data2.txt","w+")
for i in range(0,row):
    if temp[i] == 'NUC\n':
        data[i,9] = 2
        for j in range(0,10):
            if j < 9:
                f2.write("%f," % (data[i,j]))
            else:
                f2.write("%f\n" % (data[i,j]))
f2.close()
'''
ones = 0
twos = 0
for i in range(0,row):
    if data[i,9] == 1:
        ones = ones + 1
    elif data[i,9] == 2:
        twos = twos + 1
ROW = ones + twos
    

'''
    