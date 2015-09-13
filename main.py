import numpy as np
import pandas as pd
import multiprocessing
from  multiprocessing import  Pool
import feature
import logging
import time

if __name__ =="__main__":
    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    begin_time = time.time()
    multiprocessing.freeze_support()
    result = []
    p = Pool()
    result.append(p.apply_async(feature.content_basic_feature()))
    result.append(p.apply_async(feature.time_feature()))
    result.append(p.apply_async(feature.user_basic_feature()))
    p.close()
    p.join()
    end_time = time.time()
    logging.info(end_time - begin_time)