#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from collections import deque
from numba import jit,int32,guvectorize

def sentiment_feature(basic_train,basic_test):

    @guvectorize([(int64[:] )], '(n),()->(n)')
    def compute_scores(x):
        if x['极性'] =='0' or x['极性'] =='3':
            return 0
        flag =(1 if x['极性']=='1' else -1)
        return  flag * int(x['强度'])
    @vectorize([int32()])
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
