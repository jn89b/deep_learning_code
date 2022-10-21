import os
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


if __name__== '__main__':
    data = pd.read_csv('bodyfat.csv')
    data.drop(columns = ['Density'],inplace = True)
    
    data = data[['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 
                 'Hip','Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist','BodyFat']]
    
    data.head()
    
    
    X = data[['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip',
           'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']]
    y = data[['BodyFat']]
    
    training_ratio = 0.8   
    testing_ratio = 1 - training_ratio        
    validation_ratio = 0.1 * training_ratio   # validation set is subset of the training data
        
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size = testing_ratio)
    
    number_of_hidden_neurons = 15   

    model = Sequential()
    model.add(InputLayer(input_shape=(13,)))  ## 13 input features
    model.add(Dense(number_of_hidden_neurons,activation ='relu'))
    model.add(Dense(1,activation = 'linear')) ## 1 output prediction
    
    model.summary()
    
    
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', 
                      optimizer= opt,
                      metrics=[keras.metrics.RootMeanSquaredError()] 
                     )
    
    earlystop = EarlyStopping(patience= 5)
    callbacks = [earlystop]
    
    
    # Training the model
    history = model.fit(x_train, y_train,
                  batch_size= 252, ## total size length of the dataset 
                  epochs=50,
                  validation_split = validation_ratio,
                  shuffle=True,
                  callbacks = callbacks)
        
    
    colors = ['#EF7D71','#41ABD7','#36609A','#FFCE30','#194350']
    
    fig,ax  = plt.subplots(2,1, figsize =(8,8), dpi = 100)
    fig.patch.set_facecolor('#f5f6f6')
    
    axes  = ax.ravel()
    
    for ax in axes:
        ax.set_facecolor('#f5f6f6')
        for loc in ['right','top',]:
            ax.spines[loc].set_visible(False)
            
    hist1 = history.history
    Epochs =  range(len(hist1['loss']))
    
    ## loss plot
    sns.lineplot(x = Epochs, y = hist1['val_loss'],  ax = axes[0], linewidth = 2, color = colors[3])
    sns.lineplot(x = Epochs, y = hist1['loss'], ax  = axes[0], linewidth =2,  color = colors[4])
    
    
    axes[0].text(Epochs[-1]+0.25,hist1['val_loss'][-1],'Training MSE',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':colors[3]})
    axes[0].text(Epochs[-1]+0.25,hist1['loss'][-1] ,'Validation MSE',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':colors[4]})
    
    
    # accuracy plot
    sns.lineplot(x = Epochs, y = hist1['val_root_mean_squared_error'],ax = axes[1],linewidth = 2, color = colors[3])
    sns.lineplot(x = Epochs, y = hist1['root_mean_squared_error'],ax = axes[1],linewidth =2,  color = colors[4])
    axes[1].text(Epochs[-1]+0.25,hist1['root_mean_squared_error'][-1],'Validation RMSE',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':colors[4]})
    axes[1].text(Epochs[-1]+0.25,hist1['val_root_mean_squared_error'][-1] ,'Training RMSE',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':colors[3]})
    
    
    fig.text(0.1,0.98, 'Model Performance: MS and RMS loss ',{'fontfamily':'serif', 'size':18, 'weight':'bold'})

    
    train_outputs = model.predict(x_train)
    test_outputs = model.predict(x_test)

    
    colors = ['#EF7D71','#41ABD7','#36609A','#FFCE30','#194350']
    
    fig,ax  = plt.subplots(2,1, figsize =(8,8))
    fig.patch.set_facecolor('#f5f6f6')
    
    axes  = ax.ravel()
    
    for ax in axes:
        ax.set_facecolor('#f5f6f6')
        for loc in ['right','top',]:
            ax.spines[loc].set_visible(False)
            
    hist1 = history.history
    Epochs =  range(len(hist1['loss']))
    
    ## loss plot
    axes[0].scatter(x = y_train, y = train_outputs, color = colors[3])
    axes[0].set_xlabel('True Target',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':'black'})
    axes[0].set_ylabel('Output',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':'black'})
    fig.text(0.7,0.95,'Training',{'fontfamily':'serif', 'size':12, 'weight':'bold','color':colors[3]})


    # accuracy plot
    axes[1].scatter(x = y_test, y = test_outputs,color = colors[4])
    axes[1].set_xlabel('True Target',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':'black'})
    axes[1].set_ylabel('Output',{'fontfamily':'serif', 'size':11, 'weight':'bold','color':'black'})
    fig.text(0.8,0.95,'Testing',{'fontfamily':'serif', 'size':12, 'weight':'bold','color':colors[4]})
    
    
    fig.text(0.1,0.98, 'True Values and Predictions',{'fontfamily':'serif', 'size':18, 'weight':'bold'})