# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:50:28 2022

@author: jnguy

https://realpython.com/linear-regression-in-python/
Multiple parameters

Refer to help_me_1.png 
DONT DO THE BIAS TO THE SYSTEM SO THE x_0
To do :
    zip x_1k and x_2k 
    multiple x and x transpose:
        (2 x 1) (1 x 2) = (2 x 2) 
    

Can plot this as a visual element
Summation of matrices:
    https://math.stackexchange.com/questions/621036/how-sum-work-vectors-and-matrices

Inverse:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    

Find Optimal weights , W_opt for given parameters

"""



#% Import stuff here
from matplotlib import pyplot as plt



#% Classes 
import numpy as np
from sklearn.linear_model import LinearRegression

def summation_matrices():
    pass



#% Main 

if __name__ == '__main__':
    x1= [ -0.5, -0.2, -0.1, 0.3,0.4, 0.5, 0.7] 
    x2 = [ 3, 3, 2.5, 2, -1, -1, -4]
    D= [-3, -1, 0, 1.2, 1.5, 3, 4]
    
    x_list = []
    
    #constant bias = 
    
    for first,second in zip(x1,x2):
        x_list.append([first,second])    
    
    r_list = []
    for x in x_list:
        x = np.array([x])
        x_t = np.transpose(np.array(x))
        r = np.dot(x_t, x)
        r_list.append(r)
    
    R = sum(r_list)/len(r_list)
        
    p_list = []
    for d,x in zip(D,x_list):
        x = np.array([x])
        p_list.append(d*x)
    
    P = sum(p_list)/len(p_list)
    
    W_opt = np.dot(np.linalg.inv(R), np.transpose(P)) 
    
    X = np.array(x_list).transpose()
    d_est = np.dot(np.transpose(W_opt),X) #- bias 
    
    #error = abs(D - d_est)
        
        

            
    
        
    