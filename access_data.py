# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import setting
from scipy.sparse import csr_matrix


def input_data(version=0):
    '''
    读取原始文件
    return： 经过处理的dataframe
    '''

    if version ==0:
        with open(setting.raw_data_dir+'weibo_predict_data.txt')  as file:
            test  = [line.strip().split("\t") for line in file.readlines()]
        with open(setting.raw_data_dir+'weibo_train_data.txt')  as file:
            train = [line.strip().split("\t") for line in file.readlines()]
    else:
        with open(setting.raw_data_dir+'weibo_predict_data2.txt')  as file:
            test  = [line.strip().split("\t") for line in file.readlines()]
        with open(setting.raw_data_dir+'weibo_train_data2.txt')  as file:
             train = [line.strip().split("\t") for line in file.readlines()]
    test = pd.DataFrame(test)
    train = pd.DataFrame(train)
    if version ==0:
        test.loc[78336,2] = '2015-01-03'

    test[2] = pd.to_datetime(test[2])
    train[2] = pd.to_datetime(train[2])

    for i in range(3,6):
        train[i] = train[i].astype(np.int32)
    if version == 1:
        temp = train[0].copy()
        train[0] = train[1]
        train[1] = temp

        temp = test[0].copy()
        test[0] = test[1]
        test[1] = temp

    return train,test

def load_raw_data():

    train1,_ = input_data(version=0)
    train2,test2 = input_data(version=1)
    return pd.concat([train1,train2],axis=0),test2

def  save_processed_data(x,name):
    x.to_pickle(setting.processed_data_dir+name)

def load_processed_data(name):
    return pd.read_pickle(setting.processed_data_dir+name)

def output_result(result,name,docs=None):
    '''
      result: Dataframe     name: string    docs:  dict
      把dataframe转换为输出格式，保存在csv文件中
      docs为参数列表
    '''
    uid = pd.read_pickle(setting.processed_data_dir + 'uid_test')
    pid = pd.read_pickle(setting.processed_data_dir + 'pid_test')
#打印参数表
    if docs != None :
        docs = pd.Series(docs)
        docs.to_csv(setting.result_dir + name+'_parameters.txt')

    result.to_pickle(setting.result_dir + name)
    result = result.astype('int').astype('str')
    result = result.apply(lambda row:','.join([row[0],row[1],row[2]]),axis=1)

    result = pd.concat([uid,pid,result],axis = 1)
    result.to_csv(setting.result_dir + name+'.txt' ,index=False,header=False,sep='\t')

def load_lda(name,number):
    lda_result = np.load(setting.processed_data_dir+name+'.npy')
    lda_result = pd.DataFrame(lda_result,columns=['topic_%d' %i for i in range(0,number)])
    return lda_result

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])