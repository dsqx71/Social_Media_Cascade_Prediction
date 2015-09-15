__author__ = 'DongXu'
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import  KFold
import logging

def time_validation(train,time = pd.Timestamp('2014-11-15')):

    return  ((train['time'] - time) < 0).values


def user_validation(train,ratio_val= 0.2):

    index = train['uid'].unique()
    np.random.shuffle(index)
    choosen = index[:len(index)*ratio_val]
    mask = train['uid'].isin(choosen)

    return ~mask.values

def my_score_func(ground_truth,prediction):

    prediction = np.rint(prediction)
    prediction = np.where(prediction<0,0,prediction)
    deviation_f = np.abs(ground_truth[:,0]-prediction[:,0])/(5+ground_truth[:,0]).astype(np.float64)
    deviation_c = np.abs(ground_truth[:,1]-prediction[:,1])/(3+ground_truth[:,1]).astype(np.float64)
    deviation_l  = np.abs(ground_truth[:,2]-prediction[:,2])/(3+ground_truth[:,2]).astype(np.float64)
    precision = sum((ground_truth[:,0]+ground_truth[:,1] + ground_truth[:,2] +1) *  np.where(1-0.5*deviation_f - 0.25*deviation_c - 0.25*deviation_l>0.8,1,0))
    precision = precision / float(sum(ground_truth[:,0]+ground_truth[:,1] + ground_truth[:,2] +1))

    return precision

if __name__ == '__main__':

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

    train_x = np.load('processed_data/train_np.npy')
    test_x  = np.load('processed_data/test_np.npy')
    train_y = np.load('processed_data/target_np.npy')
    train_y_class = np.zeros_like(train_y)
    train_y_class[train_y[:,0]>1,0] = 1
    train_y_class[train_y[:,1]>0,1] = 1
    train_y_class[train_y[:,2]>0,2] = 1

    train_basic = pd.read_pickle('raw_data/basic_train')

    score = make_scorer(my_score_func,greater_is_better=True)
    weight = 1 + train_y[:,0] + train_y[:,1] + train_y[:,2]
    max = -1
    who = -1
    scores= []
    for max_depth  in [10,12,14,16]:
        tot = 0
        for time in [pd.Timestamp('2014-11-15'),pd.Timestamp('2014-11-30'),pd.Timestamp('2014-12-15')]:
            rf = RandomForestRegressor(n_estimators=1,max_features=35,max_depth=max_depth,n_jobs=1,verbose=2)
            mask = time_validation(train_basic,time)
            rf.fit(train_x[mask],train_y[mask],weight[mask])
            tot = tot +  score(rf,train_x[~mask],train_y[~mask])

        for i in xrange(3):
            rf = RandomForestRegressor(n_estimators=1,max_features=35,max_depth=max_depth,n_jobs=1,verbose=2)
            mask = user_validation(train_basic)
            rf.fit(train_x[mask],train_y[mask],weight[mask])
            tot = tot + score(rf,train_x[~mask],train_y[~mask])

        tot = tot /6.0

        if tot>max:
            max = tot
            who = max_depth
        scores.append((tot,max_depth))

    for i,j in scores:
        print "scores:{}   who:{}".format(i,j)

    print "max:{},who:{}".format(max,who)




