# -*- coding: UTF-8 -*-
import numpy as np
import lda
import logging
import setting
import access_data

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.info('Start to run_lda.py')

with open(setting.word_segment + 'version3_df_10.txt','r') as file:
    word = [line.decode('utf-8').split(',')[0] for line in file.readlines()]

temp = access_data.load_sparse_csr('Change_df_min/version3_df_min10.npz')
model = lda.LDA(n_topics=25, n_iter=1000, random_state=1)

x = model.fit_transform(temp)
np.save(setting.processed_data_dir + 'lda_result_version3',x)

topic_word = model.topic_word_  
np.save('topic_word_25',topic_word)

n_top_words =  20
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(word)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

