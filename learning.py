__author__ = 'DongXu'
import validation
import numpy as np
import pandas as pd
import  logging
import access_data
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

if __name__ == '__main__':

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

    logging.info('Loading data...')

    train_x = np.load('processed_data/train3_np.npy')
    test_x  = np.load('processed_data/test3_np.npy')
    train_y = np.load('processed_data/target3_np.npy')

    train_y_class = np.zeros_like(train_y)
    train_y_class[train_y[:,0]>1,0] = 1
    train_y_class[train_y[:,1]>0,1] = 1
    train_y_class[train_y[:,2]>0,2] = 1
    train_y = np.log1p(train_y)

    logging.info('Start to classify')
    clf = RandomForestClassifier(n_estimators=150,max_features=25,max_leaf_nodes=2000,n_jobs=1,verbose=2)
    clf.fit(train_x,train_y_class)
    test_predict = clf.predict(test_x)

    np.save('2015-9-24',test_predict)
    logging.info('Start to regression')
    for i in xrange(3):
        mask = train_y_class[:,i] >0
        regressor = RandomForestRegressor(n_estimators=150,max_features=75,max_leaf_nodes=2000,n_jobs=1,verbose=2)
        regressor.fit(train_x[mask],train_y[mask,i])
        test_predict[test_predict[:,i]>0,i] = np.rint(np.expm1(regressor.predict(test_x[test_predict[:,i]>0])))

    test_predict[test_predict<0] = 0
    logging.info('Start to output result')
    access_data.output_result(test_predict,'2015-9-24')
    logging.info('Finished ! ')

