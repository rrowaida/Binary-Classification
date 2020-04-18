# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:06:48 2020

@author: ragha
"""

import numpy as np
import math
import matplotlib.pyplot as plt

"""
read data function
"""
def read_data(D):
    #dat1 = np.zeros((size, d))
    #dat2 = np.zeros((size, d))
    f1=open(D, "r")    
    Row = 0
    Col = 0
    temp = 0
    for i in f1:
        for j in i:
            if ord(j) != 44 and ord(j) != 10:
                temp = temp
            else:
                temp = temp + 1                
        Row = Row + 1        
        Col = temp
        temp = 0
    size = Row
    d = Col
    Row = 0
    Col = 0
    dat1 = np.zeros((size, d))
    f1=open(D, "r") 
    for i in f1:
        num = ''
        for j in i:
            if ord(j) != 44 and ord(j) != 10:
                num = num + str(j)
            else:
                dat1[Row, Col] = float(num)            
                Col = Col + 1
                num = ''
        Row = Row + 1
        Col = 0

    
    dat1 = dat1.transpose()    
    return dat1, size, d

def Select6(Data, s, d):
    D = np.zeros((6,s))
    # Selected features 0 1 2 3 6 7
    D[0,:] = Data[0,:]
    D[1,:] = Data[1,:]
    D[2,:] = Data[2,:]
    D[3,:] = Data[3,:]
    D[4,:] = Data[6,:]
    D[5,:] = Data[7,:]
    
    return D

def ChopDat(D):
    Da = np.zeros((6,400))
    for i in range(0,400):
        Da[:,i] = D[:,i]
    return Da

def Plot_2D_data(First_X, Sec_X, First_Y, Sec_Y, fignum, dimA, dimB):
    plt.plot(First_X, Sec_X, 'ro', First_Y, Sec_Y, 'bo')
        
    minX = min(min(First_X),min(Sec_X))
    maxX = max(max(First_X),max(Sec_X))
    minY = min(min(First_Y),min(Sec_Y))
    maxY = max(max(First_Y),max(Sec_Y))
    plt.axis([minX, maxX, minY, maxY]) #sets the boundary of plot axis
    #plt.axis([-20, 20, -20, 20]) #sets the boundary of plot axis
    plt.ylabel('X ' + dimB + ' direction')
    plt.xlabel('X ' + dimA + ' direction')
    #plt.show()
    plt.savefig('Fig' + fignum + '.png')

def Plot_2D(First_X, Sec_X, First_Y, Sec_Y, dX, dY, fignum, dimA, dimB):
    plt.plot(First_X, Sec_X, 'ro', dX, dY, 'g--', First_Y, Sec_Y, 'bo')
        
    minX = min(min(First_X),min(Sec_X))
    maxX = max(max(First_X),max(Sec_X))
    minY = min(min(First_Y),min(Sec_Y))
    maxY = max(max(First_Y),max(Sec_Y))
    #plt.axis([minX, maxX, minY, maxY]) #sets the boundary of plot axis
    plt.axis([-5, 10, -10, 1.5]) #sets the boundary of plot axis
    plt.ylabel('X ' + dimB + ' direction')
    plt.xlabel('X ' + dimA + ' direction')
    #plt.show()
    plt.savefig('Fig' + fignum + '.png')
    
def get_mean(Data):
    [R,C] = Data.shape
    M = np.zeros((R,1))
    for i in range(0,R):
        M[i,0] = sum(Data[i,:])/C
    return M

def get_Sigma(data, mean):
    [R,C] = data.shape
    Sigma = np.zeros((R,R))
    for i in range(0,C):
        temp = np.zeros((R,1))
        for j in range(0,R):
            temp[j,0] = data[j,i] - mean [j, 0]
        #print(temp)
        Sigma = np.add(Sigma, np.dot(temp,temp.transpose()))
    Sigma = Sigma/C
    return Sigma

def Bayesian_mean(data, Cov, M):
    [d, s] = data.shape
    
    sig_zero = np.identity(d)
    mean = np.zeros((d,1))
    M0 = np.zeros((d,1))
    #print(mean)
    for i in range(0,d):
        M0[i,0] = data[i,0]
    
    for i in range(1,s):
        for j in range(0,i):
            for k in range(0,d):
                mean[k,0] = mean[k,0] + data[k,j]
            
        mean = mean/i
        temp = np.linalg.inv(np.add(Cov/i,sig_zero))
        
        temp2 = np.dot(np.dot(Cov, temp),M0)/i
        temp3 = np.dot(np.dot(sig_zero,temp),mean)
        #print(temp2)
        B_mean = np.add(temp2,temp3)
        
    return B_mean

def discriminate_param(M1, M2, S1, S2):
    det1 = np.linalg.det(S1)
    det2 = np.linalg.det(S2)
    
    valA = np.subtract(np.linalg.inv(S2) , np.linalg.inv(S1))
    valA = valA * 0.5    
    valB = np.subtract(np.dot(M1.transpose(),np.linalg.inv(S1)) , np.dot(M2.transpose(),np.linalg.inv(S2)))       
    valC = 0.5*np.log(det2/det1) + np.dot(np.dot(M2.transpose(),np.linalg.inv(S2)),M2) - np.dot(np.dot(M1.transpose(),np.linalg.inv(S1)),M1)
    
    return valA, valB, valC

def discriminate_check(X, valA, valB, valC, d):
    Temp_X = X
    
    return np.dot(Temp_X.transpose(),np.dot(valA,Temp_X)) + np.dot(valB.transpose(),Temp_X) + valC

def get_diag_param(D1, D2, M1, M2, Sig1, Sig2):
    
    [d, size] = D1.shape
    v1, p1 = np.linalg.eig(Sig1)
    v2, p2 = np.linalg.eig(Sig2)
    
    Trans_p1 = p1.transpose()
    Trans_p2 = p2.transpose()
    
    Y1 = np.dot(Trans_p1, Data1)
    Y2 = np.dot(Trans_p1, Data2)
    
    SigmaY1 = np.dot(np.dot(Trans_p1, Sig1),p1)
    SigmaY2 = np.dot(np.dot(Trans_p1, Sig2),p1)
    
    MeanY1 = np.dot(Trans_p1, M1)
    MeanY2 = np.dot(Trans_p1, M2)
    
    vec1inv = np.zeros((d, d))


    for i in range (0,d):
        vec1inv[i,i] = 1/math.sqrt(v1[i])
        
    
    Z1 = np.dot(vec1inv, Y1)
    Z2 = np.dot(vec1inv, Y2)
    
    SigmaZ1 = np.dot(np.dot(vec1inv, SigmaY1),vec1inv)
    SigmaZ2 = np.dot(np.dot(vec1inv, SigmaY2),vec1inv)
    
    MeanZ1 = np.dot(vec1inv, MeanY1)
    MeanZ2 = np.dot(vec1inv, MeanY2)
    
    vz2, pz2 = np.linalg.eig(SigmaZ2)    
    Trans_pz2 = pz2.transpose()  
        
    Diag = (np.dot(np.dot(Trans_pz2,vec1inv),Trans_p1))
      
    V1 = np.dot(Diag,Data1)
    V2 = np.dot(Diag,Data2)
    
    SigmaV1 = np.dot(np.dot(Trans_pz2, SigmaZ1),pz2)
    SigmaV2 = np.dot(np.dot(Trans_pz2, SigmaZ2),pz2)
    
    MeanV1 = np.dot(Trans_pz2, MeanZ1)
    MeanV2 = np.dot(Trans_pz2, MeanZ2)
    
    return V1, V2, MeanV1, MeanV2, SigmaV1, SigmaV2

def P_window(Data, dim):
    [d, s] = Data.shape
    c = Data[dim,:]
    w = (max(c) - min(c))/10
    c = np.sort(c)
    y = np.zeros(s)
    for i in range(0,s):        
        for j in range(0,s):                          
           y[i] = y[i] + (1/(w*math.sqrt(2*3.1416)))*math.exp((-0.5)*math.pow(((c[i]-c[j])/w),2))
        y[i] = y[i]/s
    plt.plot(c, y, 'r--')
    plt.axis([c[0], c[s-1], 0, 4+2]) #sets the boundary of plot axis
    plt.xlabel('value of data input n')
    plt.ylabel('probability')
    plt.show()
    
    #return y

def crossNfold(Data1, Data2, S1, S2, M1, M2, n):
    
    [Row, Col] = Data1.shape
    fold_size = int(Col/n)
    total_efficiency = 0
    
    for i in range(0,n):
        Data_copy1 = Data1
        Data_copy2 = Data2
        st = i*fold_size
        nd = (i+1)*fold_size
        #print(st,nd)
        
        Ts1 = Data_copy1[:,st:nd]
        Ts2 = Data_copy2[:,st:nd]
        #print(Ts.shape)
        for j in range(st,nd):
            Data_copy1 = np.delete(Data_copy1,st,1)
            Data_copy2 = np.delete(Data_copy2,st,1)
        Tr1 = Data_copy1
        Tr2 = Data_copy2
        
        mean1 = get_mean(Tr1)
        #print(mean1)
        mean2 = get_mean(Tr2)
        #print(mean2)
        
        covar1 = get_Sigma(Tr1, mean1)
        #print(covar1)
        covar2 = get_Sigma(Tr2, mean2)
        #print(covar2) 
        
        [valA, valB, valC] = discriminate_param(mean1, mean2, covar1, covar2)
          
        
        right_1 = 0
        right_2 = 0
        for k in range(0,fold_size):
            #Temp_X = np.array([[Ts1[0,k]],[Ts1[1,k]],[Ts1[2,k]]])
            Temp_X = np.array([[Ts1[0,k]],[Ts1[1,k]],[Ts1[2,k]], [Ts1[3,k]], [Ts1[4,k]], [Ts1[5,k]]])
            check = np.dot(Temp_X.transpose(),np.dot(valA,Temp_X)) + np.dot(valB,Temp_X) + valC
            if check>0:
                right_1 = right_1 + 1
                #print(right_1)
            #Temp_X = np.array([[Ts2[0,k]],[Ts2[1,k]],[Ts2[2,k]]])
            Temp_X = np.array([[Ts2[0,k]],[Ts2[1,k]],[Ts2[2,k]], [Ts2[3,k]], [Ts2[4,k]], [Ts2[5,k]]])
            check = np.dot(Temp_X.transpose(),np.dot(valA,Temp_X)) + np.dot(valB,Temp_X) + valC
            if check<0:
                right_2 = right_2 + 1
                #print(right_2)
        #print(right_1/fold_size)
        #print(right_1/fold_size)
        efficiency =(right_1/fold_size + right_2/fold_size)/2
        #print(efficiency)
        total_efficiency = total_efficiency + efficiency
    print(total_efficiency*100/n)
    
def NNcrossNfold(Data1, Data2, n):
    
    [Row, Col] = Data1.shape
    fold_size = int(Col/n)
    total_efficiency = 0
    
    for i in range(0,n):
        Data_copy1 = Data1
        Data_copy2 = Data2
        st = i*fold_size
        nd = (i+1)*fold_size
        #print(st,nd)
        
        Ts1 = Data_copy1[:,st:nd]
        Ts2 = Data_copy2[:,st:nd]
        #print(Ts.shape)
        for j in range(st,nd):
            Data_copy1 = np.delete(Data_copy1,st,1)
            Data_copy2 = np.delete(Data_copy2,st,1)
        
        Tr1 = Data_copy1
        Tr2 = Data_copy2                
        
        right_1 = 0
        right_2 = 0
        
        for k in range(0,fold_size):
            
            dist11 = np.zeros(Col - fold_size)
            dist12 = np.zeros(Col - fold_size)
            
            dist21 = np.zeros(Col - fold_size)
            dist22 = np.zeros(Col - fold_size)
            for l in range(0,Col - fold_size):
                temp11 = np.subtract(Ts1[:,k], Tr1[:,l])
                temp111 = np.dot(temp11.transpose(),temp11)
                dist11[l] = math.sqrt(temp111)
                
                temp12 = np.subtract(Ts1[:,k], Tr2[:,l])
                temp122 = np.dot(temp12.transpose(),temp12)
                dist12[l] = math.sqrt(temp122)
                
                temp21 = np.subtract(Ts2[:,k], Tr1[:,l])
                temp211 = np.dot(temp21.transpose(),temp21)
                dist21[l] = math.sqrt(temp211)
                
                temp22 = np.subtract(Ts2[:,k], Tr2[:,l])
                temp222 = np.dot(temp22.transpose(),temp22)
                dist22[l] = math.sqrt(temp222)
                
            dist11 = np.sort(dist11)
            dist12 = np.sort(dist12)
            
            dist21 = np.sort(dist21)
            dist22 = np.sort(dist22)
            
            dtype1 = [('source', int), ('val', float)]
            values1 = [(1, dist11[0]), (1, dist11[2]), (1, dist11[2]), (0, dist12[0]), (0, dist12[1]), (0, dist12[2])]
            a1 = np.array(values1, dtype=dtype1)       # create a structured array
            a11 = np.sort(a1, order='val')
            
            if (a11[0][0]+a11[1][0]+a11[2][0]) > 1:
            #if (a11[0][0]) == 1:
                right_1 = right_1 + 1
                
            dtype2 = [('source', int), ('val', float)]
            values2 = [(1, dist22[0]), (1, dist22[2]), (1, dist22[2]), (0, dist21[0]), (0, dist21[1]), (0, dist21[2])]
            b2 = np.array(values2, dtype=dtype2)       # create a structured array
            b22 = np.sort(b2, order='val')
            if (b22[0][0]+b22[1][0]+b22[2][0]) > 1:
            #if (b22[0][0]) == 1:
                right_2 = right_2 + 1
                
        efficiency = ((right_1/fold_size) + (right_2/fold_size))/2
        total_efficiency = total_efficiency + efficiency
        
    print(total_efficiency*100/n)
    
def Lin1crossNfold(Data1, Data2, n):
    
    [Row, Col] = Data1.shape
    fold_size = int(Col/n)
    total_efficiency = 0
    
    for i in range(0,n):
        Data_copy1 = Data1
        Data_copy2 = Data2
        st = i*fold_size
        nd = (i+1)*fold_size
        #print(st,nd)
        
        Ts1 = Data_copy1[:,st:nd]
        Ts2 = Data_copy2[:,st:nd]
        #print(Ts.shape)
        for j in range(st,nd):
            Data_copy1 = np.delete(Data_copy1,st,1)
            Data_copy2 = np.delete(Data_copy2,st,1)
        
        Tr1 = Data_copy1
        Tr2 = Data_copy2    

        [d,tr_num] = Tr1.shape            
        
        right_1 = 0
        right_2 = 0
        
        w0 = np.zeros((d+1,1))
        w1 = np.zeros((d+1,1))
        
        alph = 1
        gama = 0.1
    
        ones = np.zeros((1,tr_num)) + 1
        
        x1 = np.append(ones,Tr1,0)
        x2 = np.append(ones,Tr2,0)
        
        err = 1
        while err == 1:
            for i in range(0,tr_num):
                err = 0
                if (np.dot(w1.transpose(),x1[:,i])+gama) > (np.dot(w0.transpose(),x1[:,i])):    
                    w1 = np.subtract(w1.reshape((d+1,1)),alph*x1[:,i].reshape((d+1,1)))
                    w0 = np.add(w0.reshape((d+1,1)),alph*x1[:,i].reshape((d+1,1)))
                    err =  1
                #print(w0,'\n',w1)   
                    
                if (np.dot(w0.transpose(),x2[:,i])+gama) > (np.dot(w1.transpose(),x2[:,i])):
                    w0 = np.subtract(w0.reshape((d+1,1)),alph*x2[:,i].reshape((d+1,1)))
                    w1 = np.add(w1.reshape((d+1,1)),alph*x2[:,i].reshape((d+1,1)))
                    err = 1
        print(w0,'\n',w1)
                    
        [d,Ts_num] = Ts1.shape
        
        for i in range(0,Ts_num):           
                
            x = Ts1[:,i]
            x = np.append([1],x)
            if np.dot(w0.transpose(),x.reshape((d+1,1))) - np.dot(w1.transpose(),x.reshape((d+1,1))) > 0:
                right_1 = right_1 + 1
            x = Ts2[:,i]
            x = np.append([1],x)
            if np.dot(w0.transpose(),x.reshape((d+1,1))) - np.dot(w1.transpose(),x.reshape((d+1,1))) < 0:
                right_2 = right_2 + 1                        
              
        efficiency = ((right_1/fold_size) + (right_2/fold_size))/2
        #print(efficiency)
        total_efficiency = total_efficiency + efficiency
        
    print(total_efficiency*100/n)
    
def lin_disc1(x1, x2):
    
    [d, s] = x1.shape
    w0 = np.zeros((d+1,1))
    w1 = np.zeros((d+1,1))
    
    alph = .25
    gama = 0.1

    ones = np.zeros((1,s)) + 1
    
    x1 = np.append(ones,x1,0)
    x2 = np.append(ones,x2,0)
    
    err = 1
    while err == 1:
        for i in range(0,s):
            err = 0
            if (np.dot(w1.transpose(),x1[:,i])+gama) > (np.dot(w0.transpose(),x1[:,i])):    
                w1 = np.subtract(w1.reshape((d+1,1)),alph*x1[:,i].reshape((d+1,1)))
                w0 = np.add(w0.reshape((d+1,1)),alph*x1[:,i].reshape((d+1,1)))
                err =  1
            #print(w0,'\n',w1)   
                
            if (np.dot(w0.transpose(),x2[:,i])+gama) > (np.dot(w1.transpose(),x2[:,i])):
                w0 = np.subtract(w0.reshape((d+1,1)),alph*x2[:,i].reshape((d+1,1)))
                w1 = np.add(w1.reshape((d+1,1)),alph*x2[:,i].reshape((d+1,1)))
                err = 1
    #print(w0,'\n',w1)
    disc_x = []
    disc_y = []            
    
    for i in np.arange(-10,15,0.1):
        for j in np.arange(-10,15,0.1):
            for k in np.arange(0,1,0.1):
                x = np.array([[1],[i],[j],[0]])
                if abs(np.dot(w0.transpose(),x) - np.dot(w1.transpose(),x)) < 0.1:
                    disc_x.append(i)
                    disc_y.append(j)
                 
    #print(disc_x,disc_y)
    
    plt.plot(x1[1,:], x1[2,:],'ro', disc_x, disc_y, 'g--', x2[1,:], x2[2,:], 'bo')
    #plt.plot(x1[1,:], x1[2,:],'ro', x2[1,:], x2[2,:], 'bo')
    plt.axis([-20, 20, -20, 20]) #sets the boundary of plot axis
    plt.ylabel('X1 direction')
    plt.xlabel('X2 direction')
    plt.show()
    
def Lin2crossNfold(Data1, Data2, n):
    
    [Row, Col] = Data1.shape
    fold_size = int(Col/n)
    total_efficiency = 0
    
    for i in range(0,n):
        Data_copy1 = Data1
        Data_copy2 = Data2
        st = i*fold_size
        nd = (i+1)*fold_size
        #print(st,nd)
        
        Ts1 = Data_copy1[:,st:nd]
        Ts2 = Data_copy2[:,st:nd]
        #print(Ts.shape)
        for j in range(st,nd):
            Data_copy1 = np.delete(Data_copy1,st,1)
            Data_copy2 = np.delete(Data_copy2,st,1)
        
        Tr1 = Data_copy1
        Tr2 = Data_copy2    

        [d,tr_num] = Tr1.shape            
        
        right_1 = 0
        right_2 = 0
        
        w0 = np.zeros((d+1,1))
        alph = .25
        
        ones = np.zeros((tr_num,1)) + 1
        minus_ones = ones * (-1)
        
        x1 = Tr1.transpose()
        x1 = np.append(x1, ones, 1)
        
        x2 = Tr2.transpose()
        x2 = x2 * (-1)
        x2 = np.append(x2, minus_ones,1)
        
        X = np.append(x1, x2, 0)
        
        e = 10
        B = np.zeros((2*tr_num,1)) + .1
        Wk = w0
        
        while e > 0.000001:
            JB_k_1 = np.dot(np.subtract(np.dot(X,Wk), B).transpose(), np.subtract(np.dot(X,Wk), B))
            Wk = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), B)    
            Ek = np.subtract(np.dot(X,Wk),B)
            temp = np.zeros((2*tr_num,1))
            for i in range(0,2*tr_num):
                temp[i,0] = (Ek[i,0] + abs(Ek[i,0]))*alph
             
            B = np.add(B,temp)
            JB_k = np.dot(np.subtract(np.dot(X,Wk), B).transpose(), np.subtract(np.dot(X,Wk), B))
            e = abs(JB_k - JB_k_1)
        w0 = Wk
                    
        [d,Ts_num] = Ts1.shape
        
        for i in range(0,Ts_num):           
                
            x = Ts1[:,i]
            x = np.append(x,1)
            if np.dot(x.reshape((d+1)),w0) > 0:
                right_1 = right_1 + 1
            x = Ts2[:,i]
            x = np.append(x,1)
            if np.dot(x.reshape((d+1)),w0) < 0:
                right_2 = right_2 + 1                        
              
        efficiency = ((right_1/fold_size) + (right_2/fold_size))/2
        #print(efficiency)
        total_efficiency = total_efficiency + efficiency
        
    print(total_efficiency*100/n)
    
def lin_disc2(x11, x22):
    
    [d, s] = x11.shape
    w0 = np.zeros((d+1,1))
    
    alph = .25
    
    ones = np.zeros((s,1)) + 1
    minus_ones = ones * (-1)
    
    x1 = x11.transpose()
    x1 = np.append(x1, ones, 1)
    
    x2 = x22.transpose()
    x2 = x2 * (-1)
    x2 = np.append(x2, minus_ones,1)
    
    X = np.append(x1, x2, 0)
    
    e = 10
    B = np.zeros((2*s,1)) + .1
    Wk = w0
    
    while e > 0.000001:
        JB_k_1 = np.dot(np.subtract(np.dot(X,Wk), B).transpose(), np.subtract(np.dot(X,Wk), B))
        Wk = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), B)    
        Ek = np.subtract(np.dot(X,Wk),B)
        temp = np.zeros((2*s,1))
        for i in range(0,2*s):
            temp[i,0] = (Ek[i,0] + abs(Ek[i,0]))*alph
         
        B = np.add(B,temp)
        JB_k = np.dot(np.subtract(np.dot(X,Wk), B).transpose(), np.subtract(np.dot(X,Wk), B))
        e = abs(JB_k - JB_k_1)
   
    disc_x = []
    disc_y = []            
    
    w0 = Wk



    for i in np.arange(-10,2,0.05):
        for j in np.arange(0,10,0.05):
            for k in np.arange(0,10,10):
                #x = np.array([[i],[j],[0],[1]])
                #x = np.array([[i],[j],[0],[0], [0], [0], [1]])
                #x = np.array([[0],[i],[j],[0], [0], [0], [1]])
                #x = np.array([[0],[0],[i],[j], [0], [0], [1]])
                #x = np.array([[0],[0],[0],[i], [j], [0], [1]])
                x = np.array([[0],[0],[0],[0], [i], [j], [1]])
                #print(abs(np.dot(x.transpose(),w0)))
                if abs(np.dot(x.transpose(),w0)) < 0.02:
                    disc_x.append(i)
                    disc_y.append(j)
                 
    print(disc_x,disc_y)
    #plt.plot(x11[0,:], x11[1,:],'ro', disc_x, disc_y, 'g--', x22[0,:], x22[1,:], 'bo')
    #plt.plot(x11[1,:], x11[2,:],'ro', disc_x, disc_y, 'g--', x22[1,:], x22[2,:], 'bo')
    #plt.plot(x11[2,:], x11[3,:],'ro', disc_x, disc_y, 'g--', x22[2,:], x22[3,:], 'bo')
    #plt.plot(x11[3,:], x11[4,:],'ro', disc_x, disc_y, 'g--', x22[3,:], x22[4,:], 'bo')
    #plt.plot(x11[4,:], x11[5,:],'ro', disc_x, disc_y, 'g--', x22[4,:], x22[5,:], 'bo')
    #plt.plot(x11[1,:], x11[2,:],'ro', x22[1,:], x22[2,:], 'bo')
    plt.axis([-10, 2, 0, 10]) #sets the boundary of plot axis
    plt.ylabel('X6 direction')
    plt.xlabel('X5 direction')
    plt.show()
    
#################################### calling starts #################################

[D1, size1, d1] = read_data('Data1.txt')
[D2, size2, d2] = read_data('Data2.txt')

#[Data1, size1, d1] = read_data('dataset1.txt')
#[Data2, size2, d2] = read_data('dataset2.txt')


Dat1 = Select6(D1, size1, d1)
Dat2 = Select6(D2, size2, d2)

Data1 = ChopDat(Dat1)
Data2 = ChopDat(Dat2)

[size, d] =Data1.shape

#Plot_2D_data(Data1[0,:],Data1[1,:], Data2[0,:], Data2[1,:], '1 input Dim 1 2', '1', '2')
#Plot_2D_data(Data1[1,:],Data1[2,:], Data2[1,:], Data2[2,:], '2 input Dim 2 3', '2', '3')
#Plot_2D_data(Data1[2,:],Data1[3,:], Data2[2,:], Data2[3,:], '3 input Dim 3 4', '3', '4')
#Plot_2D_data(Data1[3,:],Data1[4,:], Data2[3,:], Data2[4,:], '4 input Dim 4 5', '4', '5')
#Plot_2D_data(Data1[4,:],Data1[5,:], Data2[4,:], Data2[5,:], '5 input Dim 5 6', '5', '6')

#lin_disc1(Data1, Data2)
#Lin1crossNfold(Data1, Data2, 5)
#lin_disc1(Data1, Data2)

#lin_disc2(Data1, Data2)

mean1 = get_mean(Data1)
mean2 = get_mean(Data2)
#print(mean1)
#print(mean2)

Sigma1 = get_Sigma(Data1, mean1)
Sigma2 = get_Sigma(Data2, mean2)
#print(Sigma1)
#print(Sigma2)

BM1 = Bayesian_mean(Data1, Sigma1, mean1)
BM2 = Bayesian_mean(Data2, Sigma2, mean2)
#print(BM1,BM2)

#crossNfold(Data1, Data2, Sigma1, Sigma2, mean1, mean2, 5)
#crossNfold(Data1, Data2, Sigma1, Sigma2, BM1, BM2, 5)

#NNcrossNfold(Data1, Data2, 5)

#Lin1crossNfold(Data1, Data2, 5)
#lin_disc1(Data1, Data2)
#Lin2crossNfold(Data1, Data2, 5)

#P_window(Data1, 0)
#P_window(Data2, 0)

#P_window(Data1, 1)
#P_window(Data2, 1)

#P_window(Data1, 2)
#P_window(Data2, 2)

#P_window(Data1, 3)
#P_window(Data2, 3)

#P_window(Data1, 4)
#P_window(Data2, 4)

#P_window(Data1, 5)
#P_window(Data2, 5)


[A, B, C] = discriminate_param(mean1, mean2, Sigma1, Sigma2)
'''
disc_x = []
disc_y = []
k = 0
for i in np.arange(0,1,0.01):
    for j in np.arange(0,1,0.01):
        #Temp_X = np.array([[i],[j],[k]])
        Temp_X = np.array([[i],[j],[k],[k],[k],[k]])
        check = np.dot(Temp_X.transpose(),np.dot(A,Temp_X)) + np.dot(B,Temp_X) + C
        #print(check)
        if abs(check) < 45:
            disc_x.append(i)
            disc_y.append(j)
print(disc_x,disc_y)

'''
disc_x = []
disc_y = []
k = 0
for i in np.arange(-10,10,0.1):
    for j in np.arange(-10,10,0.1):
        #Temp_X = np.array([[i],[j],[k]])
        Temp_X = np.array([[k],[i],[j],[k],[k],[k]])
        check = np.dot(Temp_X.transpose(),np.dot(A,Temp_X)) + np.dot(B,Temp_X) + C
        #print(check)
        if abs(check) < 5:
            disc_x.append(i)
            disc_y.append(j)
#print(disc_x,disc_y)

#print(A,B,C)            
#Plot_2D(Data1[0,:],Data1[1,:], Data2[0,:], Data2[1,:], disc_x, disc_y, '18 Discriminant function Dim 1 2', '1', '2')
#Plot_2D(Data1[1,:],Data1[2,:], Data2[1,:], Data2[2,:], disc_x, disc_y, '19 Discriminant function Dim 2 3', '2', '3')
    
#print(Sigma1)
#print(Sigma2)

[Diag_D1, Diag_D2, Diag_M1, Diag_M2, Diag_Sig1, Diag_Sig2] = get_diag_param(Data1, Data2, mean1, mean2, Sigma1, Sigma2)

[A, B, C] = discriminate_param(Diag_M1, Diag_M2, Diag_Sig1, Diag_Sig2)


disc_x = []
disc_y = []
k = 0
for i in np.arange(0,1,0.01):
    for j in np.arange(0,1,0.01):
        #Temp_X = np.array([[i],[j],[k]])
        Temp_X = np.array([[i],[j],[k],[k],[k],[k]])
        check = np.dot(Temp_X.transpose(),np.dot(A,Temp_X)) + np.dot(B,Temp_X) + C
        #print(check)
        if abs(check) < 75:
            disc_x.append(i)
            disc_y.append(j)
print(disc_x,disc_y)
'''

disc_x = []
disc_y = []
k = 0
for i in np.arange(-10,10,0.1):
    for j in np.arange(-10,10,0.1):
        Temp_X = np.array([[i],[j],[k]])
        #Temp_X = np.array([[i],[j],[k],[k],[k],[k]])
        check = np.dot(Temp_X.transpose(),np.dot(A,Temp_X)) + np.dot(B,Temp_X) + C
        #print(check)
        if abs(check) < .1:
            disc_x.append(i)
            disc_y.append(j)
print(disc_x,disc_y)
'''
#print(A,B,C)            
Plot_2D(Diag_D1[0,:],Diag_D1[1,:], Diag_D2[0,:], Diag_D2[1,:], disc_x, disc_y, '20 Diagonalized Discriminant function Dim 1 2', '1', '2')
#Plot_2D(Diag_D1[1,:],Diag_D1[2,:], Diag_D2[1,:], Diag_D2[2,:], disc_x, disc_y, '21 Diagonalised Discriminant function Dim 2 3', '2', '3')


#print(Diag_D1, Diag_D2, Diag_M1, Diag_M2, Diag_Sig1, Diag_Sig2)
#Plot_2D_data(Diag_D1[0,:],Diag_D1[1,:], Diag_D2[0,:], Diag_D2[1,:], '6 Diagonalized Dim 1 2', '1', '2')
#Plot_2D_data(Diag_D1[1,:],Diag_D1[2,:], Diag_D2[1,:], Diag_D2[2,:], '7 Diagonalized Dim 2 3', '2', '3')

#print(Diag_M1, Diag_M2)
#print(Diag_Sig1, Diag_Sig2)

#crossNfold(Diag_D1, Diag_D2, Diag_Sig1, Diag_Sig2, Diag_M1, Diag_M2, 5)
#crossNfold(Data1, Data2, Sigma1, Sigma2, BM1, BM2, 5)

#NNcrossNfold(Diag_D1, Diag_D2, 5)
#Lin1crossNfold(Diag_D1, Diag_D2, 5)
#lin_disc1(Diag_D1, Diag_D2)
#lin_disc2(Diag_D1, Diag_D2)
#Lin2crossNfold(Diag_D1, Diag_D2, 5)

