import numpy as np
import pandas as pd
import multiprocessing
from  multiprocessing import  Pool
import feature

if __name__ =="__main__":
    multiprocessing.freeze_support()
    result = []
    p = Pool(3)
    result.append(p.apply_async(feature.content_basic_feature()))
    result.append(p.apply_async(feature.time_feature()))
    result.append(p.apply_async(feature.user_basic_feature()))
    p.close()
    p.join()