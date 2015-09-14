__author__ = 'DongXu'

import numpy as np
import cPickle
from  multiprocessing import  Pool
import logging
import multiprocessing

from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor,\
RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

def clf(num, train_x,  train_y, train_y_class):
    logging.info("Start to learning clf :" + str(num))
    c = GradientBoostingClassifier(learning_rate=0.1,n_estimators=1500,subsample=0.5,max_leaf_nodes=10,max_features=50,verbose=2)
    weight = train_y +1
    c.fit(train_x,train_y_class,sample_weight=weight)
    return c,num,'clf'


def regression(num,train_x,train_y):
    logging.info("start to learning regression :" + str(num))
    regressor =GradientBoostingRegressor(learning_rate=0.1,n_estimators=1500,subsample=0.5,max_leaf_nodes= 10,max_features=50,verbose=2)
    regressor.fit(train_x,train_y,sample_weight=train_y+1)
    return regressor,num,'regression'

if __name__ == '__main__':

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    multiprocessing.freeze_support()

    train_x = np.load('processed_data/train_np.npy')
    test_x  = np.load('processed_data/test_np.npy')
    train_y = np.load('processed_data/target_np.npy')

    train_y_class = np.zeros_like(train_y)
    train_y_class[train_y[:,0]>1,0] = 1
    train_y_class[train_y[:,1]>0,1] = 1
    train_y_class[train_y[:,2]>0,2] = 1
    #train_y = np.log1p(train_y)

    p = Pool(1)
    temp = []
    for num in xrange(1):
        #temp.append(p.apply_async(clf,args=(num,train_x,train_y[:,num],train_y_class[:,num])))
        #mask = train_y_class[:,num] >0
        temp.append(p.apply_async(regression,args=(num,train_x,train_y[:,num])))
    p.close()
    p.join()
    result = []
    for i in temp:
        result.append(i.get())
    with open('result','wb') as file:
        cPickle.dump(result,file)





