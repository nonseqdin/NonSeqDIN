import pickle
import pandas as pd
import pyodps

DATASET_PKL = './data/taobao_data/dataset.pkl'
with open(DATASET_PKL, 'rb') as f:
    train_sample_list = pickle.load(f)
    test_sample_list = pickle.load(f)


columns = 'uid,is_train,query__item,query__cate,query__hist_item,query__hist_cate,label'.split(',')
D = pd.DataFrame(data=train_sample_list, columns=columns)
print(D.shape)
print(D.head())

pyodps.drop_table('s_35926_dataset_taobao_train')
pyodps.write_table(D, 's_35926_dataset_taobao_train')

D = pd.DataFrame(data=test_sample_list, columns=columns)
print(D.shape)
print(D.head())

pyodps.drop_table('s_35926_dataset_taobao_test')
pyodps.write_table(D, 's_35926_dataset_taobao_test')

pyodps.drop_table('s_35926_dataset_taobao')
pyodps.run('''
        create table s_35926_dataset_taobao as
        select * from s_35926_dataset_taobao_test
        union all
        select * from s_35926_dataset_taobao_train
        ''')
