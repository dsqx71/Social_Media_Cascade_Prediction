# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import setting
import logging
from setting import func

logging.root.setLevel(level=logging.INFO)
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

    for string in ['share','comment','zan']:
        temp = pd.DataFrame({string+'_histogram':group_train[string].agg(lambda x:np.histogram(x,bins=[-1,1,3,10,33,100,333,1000,100000]))})
        train = train.merge(temp,left_on='uid',right_index=True,suffixes=('','_'+'histogram'),how='left')
        test  =   test.merge(temp,left_on='uid',right_index=True,suffixes=('','_'+'histogram'),how='left')
    test.rename(columns={'share':'share_mean','comment':'comment_mean','zan':'zan_mean'},inplace=True)

    test.fillna(-1,inplace=True)
    train.fillna(-1,inplace=True)
#    for name in ['share','comment','zan']:
#        train[name+'_histogram'] = train[name+'_histogram'].map(lambda x: x[0] )
#        test[name+'_histogram'] = test[name+'_histogram'].map(lambda x: x[0] )
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
                              pd.DataFrame( test[['uid','content_len','链接','//@','@','#','【','《','\[']])])
    group = tot.groupby('uid')
    train = train[['uid','pid']].copy()
    test  = test[['uid','pid']].copy()

    for f in func:
       train = train.merge(group['content_len','链接','//@','@','#','【','《','\['].agg(f),left_on='uid',right_index=True,how='left',suffixes=('','_'+f.func_name))
       test  = test.merge(group['content_len','链接','//@','@','#','【','《','\['].agg(f),left_on='uid',right_index=True,how='left',suffixes=('','_'+f.func_name))
    train.rename(columns={'链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[':'\[_mean'},inplace=True)
    test.rename(columns={'链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[':'\[_mean'},inplace=True)

    for string,values in [('content_len',[0,5,10,20,50,100,200,300,430]),
                                   ('链接',[0,1,2,3,12]),
                                   ('//@',[0,1,2,15]),
                                  ('@',[0,1,2,3,32]),
                                   ('#',[0,1,2,3,4,100]),
                                   ('【',[0,1,2,3,12]),
                                   ('《',[0,1,2,3,22]),
                                   ('\[',[0,1,2,3,4,70]) ]:
        temp = pd.DataFrame({string+'_histogram':group[string].agg(lambda x:np.histogram(x,bins=values))})
        train = train.merge(temp,left_on='uid',right_index=True,how='left')
        test  = test.merge(temp,left_on='uid',right_index=True,how='left')

    for string in ['content_len','链接','//@','@','#','【','《','\[']:
        train[string+'_histogram'] = train[string +'_histogram'] .map(lambda  x:x[0])
        test[string + '_histogram']  = test[string + '_histogram'].map(lambda x:x[0])
    test.fillna(-1,inplace=True)
    train.fillna(-1,inplace=True)

    train.drop('uid',axis=1,inplace=True)
    test.drop('uid',axis = 1,inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)

    return train,test


def  time_feature():
    '''
        时间特征
    '''
    train = pd.read_pickle('raw_data/basic_train')
    test  = pd.read_pickle('raw_data/basic_test')
    train = train[['uid','pid','time','tot_counts','raw_corpus']]
    test  = test[['uid','pid','time','tot_counts','raw_corpus']]

    #星期几
    train['dayofweek']  = train.time.dt.dayofweek
    test['dayofweek']   = test.time.dt.dayofweek
    #月份
    train['month'] = train.time.dt.month
    test['month'] = 13

    tot = pd.concat([train[['uid','dayofweek','tot_counts']],test[['uid','dayofweek','tot_counts']]])
    group = tot.groupby('uid')['dayofweek']
    counts = tot.groupby('uid')['tot_counts'].agg(np.unique)

    for i in range(0,7):
        temp = pd.DataFrame({'dayofweek_'+str(i):group.agg(lambda x:sum(x==i)),'tot_counts':counts})
        temp['ratio_week'+str(i)] = temp.apply(lambda x:x['dayofweek_'+str(i)]/float(x['tot_counts']),axis=1)
        train = train.merge(temp[['dayofweek_'+str(i),'ratio_week'+str(i)]],left_on='uid',right_index=True,how='left')
        test  =  test.merge(temp[['dayofweek_'+str(i),'ratio_week'+str(i)]],left_on='uid',right_index=True,how='left')

    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)

    train['len'] = train['raw_corpus'].map(lambda x:len(x))
    test['len'] = test['raw_corpus'].map(lambda x:len(x))

    temp = pd.concat([train[['uid','time','len']],test[['uid','time','len']]])
    temp.sort(columns=['uid','time'],inplace=True)
    temp['time_diff'] = temp.time.diff()
    temp['len_diff']  = temp.len.diff()
    mask = temp.uid.shift(1) !=temp.uid
    temp.loc[mask,'time_diff'] = temp.loc[mask,'time'] - pd.Timestamp('2014-6-30')
    temp.loc[mask,'len_diff'] = 10000
    train = train.merge(temp[['time_diff','len_diff']],left_index=True,right_index=True,how='left')
    test  = test.merge(temp[['time_diff','len_diff']],left_index=True,right_index=True,how='left')

    train['time_diff'] = train['time_diff'].dt.days
    test['time_diff'] = test['time_diff'].dt.days
    train[['dayofweek','month','len','time_diff','len_diff']] = train[['dayofweek','month','len','time_diff','len_diff']].astype(np.int32)
    test[['dayofweek','month','len','time_diff','len_diff']] = test[['dayofweek','month','len','time_diff','len_diff']].astype(np.int32)
    train.drop(['raw_corpus','uid','time','tot_counts'],inplace=True,axis=1)
    test.drop(['raw_corpus','uid','time','tot_counts'],inplace=True,axis=1)

    return train,test
