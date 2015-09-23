#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from collections import deque
from numba import jit,int16
def find_latest(basic_train,basic_test):

    train  = basic_train[['uid','pid','time']].copy()
    test   =   basic_test[['uid','pid','time']].copy()
    tot = pd.concat([train,test],axis=0)
    tot.sort(columns=['uid','time'],inplace=True)
    tot['latest']  = np.zeros(tot.shape[0])
    tot.index = range(tot.shape[0])

    mask = tot['uid'] != tot['uid'].shift(-1)
    tot['latest'] = tot['time'].diff().shift(-1).dt.days
    tot[mask,'latest']  = 500
    train = train.merge(tot[['pid','latest']],left_on='pid',right_on='pid',how='left')
    test   = test.merge(tot[['pid','latest']],left_on='pid',right_on='pid',how='left')

    train.drop(['uid','time'],axis=1,inplace=True)
    test.drop(['uid','time'],axis=1,inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)
    return train,test

def find_seven_days(basic_train,basic_test):
    '''
            用队列来维护
    '''
    train  = basic_train[['uid','pid','time']].copy()
    test   = basic_test[['uid','pid','time']].copy()
    tot = pd.concat([train,test],axis=0)
    tot.sort(columns=['uid','time'],inplace=True)
    tot['seven_days']  = np.zeros(tot.shape[0])
    tot.index = range(tot.shape[0])

    queue =deque([tot.at[tot.shape[0]-1,'time']])

    for x in xrange(tot.shape[0]-2,-1,-1):
        if tot.at[x,'uid'] != tot.at[x+1,'uid']:
            queue = deque([])
        if  len(queue)>0:
            while  (queue[0]-tot.at[x,'time']).days >7:
                queue.popleft()
                if len(queue)==0:
                    break
            tot.at[x,'seven_days'] = len(queue)
        queue.append(tot.at[x,'time'])

    train = train.merge(tot[['pid','seven_days']],left_on='pid',right_on='pid',how='left')
    test   = test.merge(tot[['pid','seven_days']],left_on='pid',right_on='pid',how='left')

    train.drop(['uid','time'],axis=1,inplace=True)
    test.drop(['uid','time'],axis=1,inplace=True)
    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)
    return train,test

def sentiment_feature(basic_train,basic_test):

    @jit
    def compute_scores(x):
        if x['极性'] =='0' or x['极性'] =='3'
            return 0
        flag =(1 if x['极性']=='1' else -1)
        return  flag * int(x['强度'])

    def compute_sentiment_scores(line):
        words = np.array(line.split(' '))
        mask = sentiment['词语'].isin(words)
        if mask.any() == False :
            return 0
        else:
            return sum(sentiment[mask].apply(compute_scores,axis=1))

    with open('sentiment_word.csv') as file:
        result = [line.strip().split('\t')  for line in file.readlines()]
    for line in result:
        if len(line)>10:
            result.remove(line)
    sentiment = pd.DataFrame(columns=result[0],data=result[1:])
    sentiment['词语']=sentiment['词语'].map(lambda x:x.decode("utf-8"))
    sentiment.columns = range(10)
    sentiment.drop([7,8,9],axis=1,inplace=True)
    sentiment.columns = ['词语','词性','词义数','词义序号','分类','强度','极性']
    sentiment.drop(['词性','词义数','词义序号','分类'],axis=1,inplace=True)
    train  = basic_train[['clean&segment','pid']].copy()
    test   = basic_test[['clean&segment','pid']].copy()

    train['sentiment'] = train['clean&segment'].map(compute_sentiment_scores)
    test['sentiment']  = test['clean&segment'].map(compute_sentiment_scores)

    train.drop('clean&segment',axis=1,inplace=True)
    test.drop('clean&segment',axis=1,inplace=True)

    train.set_index('pid',inplace=True)
    test.set_index('pid',inplace=True)
    return train,test
