import pandas as pd
import numpy as np

def find_latest(basic_train,basic_test):

    train  = basic_train[['uid','pid','time']].copy()
    test   =   basic_test[['uid','pid','time']].copy()
    tot = pd.concat([train,test],axis=0)
    tot.sort(columns=['uid','time'],inplace=True)
    tot['latest']  = np.zeros(tot.shape[0])
    tot.index = range(tot.shape[0])

    mask = tot['uid'] != tot['uid'].shift(-1)
    tot['latest'] = tot['time'].diff().shift(-1).dt.days
    tot[mask,'latest']  = 365
    train = train.merge(tot[['pid','latest']],left_on='pid',right_on='pid',how='left')
    test   = test.merge(tot[['pid','latest']],left_on='pid',right_on='pid',how='left')

    train.drop(['uid','time'],axis=1,inplace=True)
    test.drop(['uid','time'],axis=1,inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)
    return train,test