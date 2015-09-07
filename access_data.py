# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import setting
from scipy.sparse import csr_matrix

def load_raw_data():
    '''
    读取原始文件
    return： 经过处理的dataframe
    '''
    with open(setting.raw_data_dir+'weibo_predict_data.txt')  as file:
        test  = [line.strip().split("\t") for line in file.readlines()]
    with open(setting.raw_data_dir+'weibo_train_data.txt')  as file:
        train = [line.strip().split("\t") for line in file.readlines()]

    test = pd.DataFrame(test)
    train = pd.DataFrame(train)
    #test.drop(78336,axis= 0,inplace=True)  # 这行的日期有错，并且content没有特别有价值的信息
    test.loc[78336,2] = '2014-12-04'
    test[2] = pd.to_datetime(test[2])
    train[2] = pd.to_datetime(train[2])
    for i in range(3,6):
        train[i] = train[i].astype(np.int8)
    return train,test

def  save_processed_data(x,name):
    x.to_pickle(setting.processed_data_dir+name)

def load_processed_data(name):
    return pd.read_pickle(setting.processed_data_dir+name)

def output_result(result,name,docs=None):
    '''
      result: Dataframe     name: string    docs:  dict
    '''
    uid = pd.read_pickle(setting.processed_data_dir + 'uid_test')
    pid = pd.read_pickle(setting.processed_data_dir + 'pid_test')

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