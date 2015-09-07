# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import setting
import logging

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.info('Start to get features')

def produce_statistical_feature():
    '''
            根据基本的特征进行扩展,增加统计特征，min，max，std，histogram等
    '''
    train = pd.read_pickle(setting.raw_data_dir + 'basic_train')
    test = pd.read_pickle(setting.raw_data_dir + 'basic_test')

    func = [np.mean,np.max,np.min,np.std]
    group_train = train.groupby('uid')

    '''
        用户在训练集中的share，comment，zan的统计特征，由于测试集中许多用户没有出现在训练集中，所以test中有许多缺失值
        测试集中900+用户，800+不在训练集中,所以仅靠share，comment，zan没有什么用。
    '''
    for f in func:
        train = train.merge(group_train['share','comment','zan'].agg(f),left_on='uid',right_index=True,suffixes=('','_'+f.func_name),how='left')
        test  = test.merge(group_train['share','comment','zan'].agg(f),left_on='uid',right_index=True,suffixes=('','_'+f.func_name),how='left')

    test.rename(columns={'share':'share_mean','comment':'comment_mean','zan':'zan_mean'},inplace=True)
#    train.fillna({'share_std':1.89*1.5,'comment_std':1.19*1.5,'zan_std':0.588*1.5},inplace=True)
#   test.fillna({'share_std':1.89*1.5,'comment_std':1.19*1.5,'zan_std':0.588*1.5},inplace=True)
    test.fillna(-1,inplace=True)

    #在training set和test set中和用户发送微博的总数量
    tot = pd.concat([pd.DataFrame(train['uid']),pd.DataFrame(test['uid'])])
    c = pd.DataFrame(tot['uid'].value_counts())
    c.columns = ['tot_counts']

    train = train.merge(c,left_on='uid',right_index=True,how='left')
    test  = test.merge(c,left_on='uid',right_index=True,how='left')

    # 用户出现在训练集的次数
    c = pd.DataFrame(train['uid'].value_counts())
    c.columns = ['train_counts']

    train = train.merge(c,left_on='uid',right_index=True,how='left')
    test  = test.merge(c,left_on='uid',right_index=True,how='left')

    #统计文本长度
    train['content_len'] = train['raw_corpus'].map(lambda x: len(x))
    test['content_len']  = test['raw_corpus'].map(lambda x:len(x))


    return train,test
