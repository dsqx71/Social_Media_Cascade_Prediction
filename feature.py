__author__ = 'DongXu'
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import setting

def apply_group():

def produce_statistical_feature():

    train_basic = pd.read_pickle(setting.raw_data_dir + 'basic_train')
    test_basic = pd.read_pickle(setting.raw_data_dir + 'basic_test')

    func = [np.mean,np.max,np.min,np.std]
    for f in func:


'''
    train = train.merge(t.groupby('uid').mean(),left_on='uid',right_index=True,suffixes=('','_mean'),how='left')
    test  = test.merge(t.groupby('uid').mean(),left_on='uid',right_index=True,suffixes=('','_mean'),how='left')

train = train.merge(t.groupby('uid')['share','comment','zan'].max(),left_on='uid',right_index=True,suffixes=('','_max'),how='left')
test  = test.merge(t.groupby('uid')['share','comment','zan'].max(),left_on='uid',right_index=True,suffixes=('','_max'),how='left')

train = train.merge(t.groupby('uid')['share','comment','zan'].min(),left_on='uid',right_index=True,suffixes=('','_min'),how='left')
test  = test.merge(t.groupby('uid')['share','comment','zan'].min(),left_on='uid',right_index=True,suffixes=('','_min'),how='left')

train = train.merge(t.groupby('uid')['share','comment','zan'].std(),left_on='uid',right_index=True,suffixes=('','_std'),how='left')
test  = test.merge(t.groupby('uid')['share','comment','zan'].std(),left_on='uid',right_index=True,suffixes=('','_std'),how='left')

train.fillna({'share_std':1.89*1.5,'comment_std':1.19*1.5,'zan_std':0.588*1.5},inplace=True)
test.fillna({'share_std':1.89*1.5,'comment_std':1.19*1.5,'zan_std':0.588*1.5},inplace=True)
'''