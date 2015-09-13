# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

from setting import func

def  user_basic_feature(basic_train,basic_test):
    '''
            根据基本的特征进行扩展,增加统计特征，min，max，std，histogram等
    '''
    train = basic_train[['uid','pid','share','comment','zan']].copy()
    test =    basic_test[['uid','pid']].copy()

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

    train[['share_mean','comment_mean','zan_mean','share_std','comment_std','zan_std']] =  train[['share_mean','comment_mean','zan_mean','share_std','comment_std','zan_std']] .astype(np.float32)
    test[['share_mean','comment_mean','zan_mean','share_std','comment_std','zan_std']]  = test[['share_mean','comment_mean','zan_mean','share_std','comment_std','zan_std']].astype(np.float32)

    return train,test

def  content_basic_feature(basic_train,basic_test):
    '''
        文本长度和符号的统计量
    '''
    train  = basic_train[['uid','pid','raw_corpus','链接','//@','@','#','【','《','\[']].copy()
    test   = basic_test[['uid','pid','raw_corpus','链接','//@','@','#','【','《','\[']].copy()
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
    train.rename(columns={'content_len':'content_len_mean','链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[':'\[_mean'},inplace=True)
    test.rename(columns={'content_len':'content_len_mean','链接':'链接_mean','//@':'//@_mean','@':'@_mean','#':'#_mean','【':'【_mean','《':'《_mean','\[':'\[_mean'},inplace=True)

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
    test.drop('uid',axis =1,inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)

    for string  in ['content_len','链接','//@','@','#','【','《','\[']:
        train[string+"_mean"] = train[string+"_mean"] .astype(np.float32)
        train[string+"_std"] = train[string+"_std"].astype(np.float32)
        test[string+"_mean"] = test[string+"_mean"] .astype(np.float32)
        test[string+"_std"] = test[string+"_std"].astype(np.float32)

    return train,test


def  time_feature(basic_train,basic_test):
    '''
        时间特征
    '''
    train = basic_train[['uid','pid','time','tot_counts','raw_corpus']].copy()
    test  = basic_test[['uid','pid','time','tot_counts','raw_corpus']].copy()

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
