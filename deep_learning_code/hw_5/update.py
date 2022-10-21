# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:02:13 2022

@author: jnguy
"""

import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt

#% Import stuff here
def update_weights(weights:list,alpha:float, error:float):
    pass

def relu_evaluate(y_predict) -> int:
    """relu function evaluate"""
    if y_predict >= 0.5:
        return 1
    else:
        return 0
    

def is_val_correct(y_true, y_pred):
    
    if y_true != y_pred:
        return False
    
    return True

#% Classes 



#% Main 
if __name__ =='__main__':
    plt.close('all')
    
    X = np.array([[1,-1,1],[1,-1, -0.5],[1,-0.5, 0.5],[1,0, -0.5],[1,1, 1],[1,1, 0],[1,1,-0.5],[1,0, 1]])
    
    Y = np.array([
        [0],
         [0],
         [0],
         [0],
         [1],
         [1],
         [1],
         [1]])
    
    
    # w0 = 0.5
    # w1 = 3.5
    # w2 = 1
    
    # w0 = 0
    # w1 = 4.5
    # w2 = 3.5
    
    w0 = 2
    w1 = 4.5
    w2 = 4.5
    
    w0 = 1.15
    w1 = 5.35
    w2 = 3.65
    
    
    w_list = np.array([[w0 , w1,w2]])
    
    result_list = []
    
    done =  False
    
    overall_list = []

        
    guess_list = []
    
    alpha = 0.85
    
    for i, x in enumerate(X):            
        
        y_val = np.dot(w_list, x.transpose())
        y_predict = relu_evaluate(y_val)
        print(y_val[0], y_predict, Y[i])
        
        error = Y[i]- y_predict
        print("error is ", error, Y[i], y_predict)
        print("wlist ", w_list)
        
        # w_list[0][0] =  w_list[0][0] + alpha*error * x[0]
        # w_list[0][1] =  w_list[0][1] + alpha*error * x[1]
        # w_list[0][2] =  w_list[0][2] + alpha*error * x[2]
        
        
        
    # for i, x in enumerate(X):
    #     print(i)
    #     y_val = np.dot(w_list, x.transpose())
    #     y_predict = relu_evaluate(y_val)
    #     print("predicted ", y_predict, Y[i][0])
    
    t = np.linspace(-1,1)
    m = -w_list[0][1]/w_list[0][2]
    b = w_list[0][0]
    b = 0.3
    y_line = b + m*t
    plt.plot(t, y_line)
    plt.scatter(X[:,1], X[:,2])
            
    
    



