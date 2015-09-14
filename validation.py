__author__ = 'DongXu'
import numpy as np
import pandas as pd

def time_validation(time = pd.Timestamp('2014-11-15'),train_x,train_y,train):
    mask = (train['time']  -time) < 0
    x = train_x[mask]
    y = train_y[mask]
    x_val = train_x[~mask]
    y_val = train_y[~mask]
    return x,y,x_val,y_val

def user_validation(ratio_val= 0.1,train_x,train_y,train):
    index = train['uid'].unique()
    np.random.shuffle(index)
    choosen = index[:len(index)*ratio_val]
    mask = train['uid'].isin(choosen)
    return train_x[~mask],train_y[~mask],train_x[mask],train_y[mask]

def crossValidation(fold = 3,train_x,train_y):
