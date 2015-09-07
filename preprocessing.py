# -*- coding: UTF-8 -*-
__author__ = 'DongXu'
import numpy as np
import pandas as pd
import access_data
import re
import setting
import logging
import jieba
from sklearn.preprocessing import LabelEncoder
import os.path
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

def clean_corpus():
    '''
       清理原始语聊：并且提取符号特征
    '''
    train , test = access_data.load_raw_data()
    train = pd.DataFrame({'corpus' :train[6]})
    test  = pd.DataFrame({'corpus' : test[3]})

    keyword = [r'http[0-9a-zA-Z?:=._@%/\-#&\+|]+' ,r'//@',   r'@' ,  r'#' ,  r'【' ,r'《' ,r'\[' ]

    for string in keyword:
        reg = re.compile(string)
        train[string] = train['corpus'].map(lambda x : reg.subn(' ',x))
        test[string] = test['corpus'].map(lambda x : reg.subn(' ',x))

        train['corpus']  = train[string].map(lambda x:x[0])
        test['corpus']   = test[string].map(lambda x:x[0])

        train[string] = train[string].map(lambda x:x[1])
        test[string]  = test[string].map(lambda x:x[1])

        logging.info('finished cleaning symbol %s' % string)

    return train,test

def segment_word(raw_data):
    '''
    raw_data: 对原始语料进行分词
    return: 已经分词了的语料
    对清理后的文本进行分词
    '''
    with open('stop.txt','r') as file:
        stop = [word.strip().decode('utf-8') for word in file.readlines()]

    def cut_word(x):
        t = []
        for i in xrange(len(x)):
            temp = jieba.cut(x[i],cut_all=False)
            temp = set(temp)-set(stop)
            t.append(' '.join(list(temp)))
            if  i % 100000 == 0:
                logging.info('have segmented %d' % i)
        t = pd.DataFrame(t)
        return t

    logging.info('Start to segment')
    return cut_word(raw_data)

def encode_label():
    '''
        给对原始的uid 排序，得到有序的pid
    '''
    train,test = access_data.load_raw_data()
    uid_train = train[0].values
    uid_test  = test[0].values

    labelencoder = LabelEncoder()
    labelencoder.fit(list(uid_train)+list(uid_test))

    uid_new_train = labelencoder.transform(uid_train)
    uid_new_test  = labelencoder.transform(uid_test)

    x ,y = pd.DataFrame({'uid':uid_new_train,'pid':train[1].values}) , pd.DataFrame({'uid':uid_new_test,'pid':test[1].values})

    addr1 = setting.processed_data_dir + 'uid&pid_train'
    addr2 = setting.processed_data_dir + 'uid&pid_test'
    if not os.path.exists(addr1):
        x.to_pickle(addr1)

    if not os.path.exists(addr2):
        y.to_pickle(addr2)
    return x,y

def bag_of_word(x,y,min_df=15):
    '''
        计算词的频率
    '''
    vectorizer = CountVectorizer(min_df= min_df ,lowercase=True,stop_words='english',dtype=np.int32)

    temp = vectorizer.fit_transform(np.r_[x,y])
    access_data.save_sparse_csr('Change_df_min/version2_df_min%d'%min_df,temp)

    vacabulary = pd.Series(vectorizer.vocabulary_)
    vacabulary.to_csv('Change_df_min/version2_df_%d.txt'% min_df,encoding='utf-8')














