# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import setting
import logging
from setting import func

logging.root.setLevel(level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

def  user_basic_feature():
    '''
            根据基本的特征进行扩展,增加统计特征，min，max，std，histogram等
    '''
    train = pd.read_pickle(setting.raw_data_dir + 'basic_train')
    test = pd.read_pickle(setting.raw_data_dir + 'basic_test')

#  计算share、comment、zan的统计量

    group_train = train.groupby('uid')
    for f in func:
        train = train.merge(group_train['share','comment','zan'].agg(f),left_on='uid',right_index=True,suffixes=('','_'+f.func_name),how='left')
        test  = test.merge(group_train['share','comment','zan'].agg(f),left_on='uid',right_index=True,suffixes=('','_'+f.func_name),how='left')
    train = train.merge(group_train['share','comment','zan'].agg(lambda x:np.histogram(x,bins=[0,1,3,10,33,100,333,1000,100000])),left_on='uid',right_index=True,suffixes=('','_'+'histogram'),how='left')
    test  = test.merge(group_train['share','comment','zan'].agg(lambda x:np.histogram(x,bins=[0,1,3,10,33,100,333,1000,100000])),left_on='uid',right_index=True,suffixes=('','_'+'histogram'),how='left')

    for string in ['share','comment','zan']:
        temp = string + '_histogram'
        train[temp] = train[temp].map(lambda x:x[0])
        test[temp]  = test[temp].map(lambda x:x[0])

    test.rename(columns={'share':'share_mean','comment':'comment_mean','zan':'zan_mean'},inplace=True)

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

    test.fillna(-1,inplace=True)
    train.fillna(-1,inplace=True)

    return train,test

def  content_basic_feature():
    '''
        文本长度和符号的统计量
    '''
    train = pd.read_pickle(setting.raw_data_dir + 'basic_train')
    test = pd.read_pickle(setting.raw_data_dir + 'basic_test')

    #统计文本长度
    train['content_len'] = train['raw_corpus'].map(lambda x:len(x))
    test['content_len']  = test['raw_corpus'].map(lambda x:len(x))

    #文本的统计量
    tot = pd.concat([pd.DataFrame(train[['uid','content_len','链接','//@','@','#','【','《','\[']]),\
                              pd.DataFrame(test[['uid','content_len','链接','//@','@','#','【','《','\[']])])
    group = tot.groupby('uid')

    train = train[['uid','pid']].copy()
    test  = test[['uid','pid']].copy()
    for f in func:
       train = train.merge(group['content_len','链接','//@','@','#','【','《','\['].agg(f),left_on='uid',right_index=True,how='left',suffixes=('','_'+f.func_name))
       test  = test.merge(group['content_len','链接','//@','@','#','【','《','\['].agg(f),left_on='uid',right_index=True,how='left',suffixes=('','_'+f.func_name))

    train.rename(columns={'链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[_mean'})
    test.rename(columns={'链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[_mean'})
    for string,values in [('content_len',[0,5,10,20,50,100,200,300,430]),
                                   ('链接',[0,1,2,3,12]),
                                   ('//@',[0,1,2,15]),
                                  ('@',[0,1,2,3,32]),
                                   ('#',[0,1,2,3,4,100]),
                                   ('【',[0,1,2,3,12]),
                                   ('《',[0,1,2,3,22]),
                                   ('\[',[0,1,2,3,4,70])
                                    ]:
        temp = pd.DataFrame({string+'_histogram':group[string].agg(lambda x:np.histogram(x,bins=values))})
        train = train.merge(temp,left_on='uid',right_index=True,how='left')
        test  = test.merge(temp,left_on='uid',right_index=True,how='left')

    for string in ['content_len','链接','//@','@','#','【','《','\[']:
        train[string +'_histogram'] = train[string +'_histogram'] .map(lambda  x:x[0])
        test[string + '_histogram']  = test[string + '_histogram'].map(lambda x:x[0])
    test.fillna(-1,inplace=True)
    train.fillna(-1,inplace=True)

    train.drop(['uid'],inplace=True)
    test.drop(['uid'],inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)

    return train,test
