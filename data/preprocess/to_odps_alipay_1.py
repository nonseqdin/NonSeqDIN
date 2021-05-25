import pickle
import pandas as pd
import pyodps

DATASET_PKL = './data/alipay_data/dataset.pkl'
with open(DATASET_PKL, 'rb') as f:
    train_sample_list = pickle.load(f)
    valid_sample_list = pickle.load(f)
    test_sample_list = pickle.load(f)


columns = 'query__uid,query__item,query__cate,query__sid,query__hist_item,query__hist_cate,query__hist_sid,label'
columns = columns.split(',')

D = pd.DataFrame(data=train_sample_list, columns=columns)
print(D.shape)
print(D.head())

pyodps.drop_table('s_35926_dataset_alipay_train')
pyodps.write_table(D, 's_35926_dataset_alipay_train')

D = pd.DataFrame(data=valid_sample_list, columns=columns)
print(D.shape)
print(D.head())

pyodps.drop_table('s_35926_dataset_alipay_valid')
pyodps.write_table(D, 's_35926_dataset_alipay_valid')

D = pd.DataFrame(data=test_sample_list, columns=columns)
print(D.shape)
print(D.head())

pyodps.drop_table('s_35926_dataset_alipay_test')
pyodps.write_table(D, 's_35926_dataset_alipay_test')


pyodps.drop_table('s_35926_dataset_alipay')
pyodps.run('''
        create table s_35926_dataset_alipay lifecycle 90 as
        select 0 is_train, * from s_35926_dataset_alipay_test
        union all
        select 2 is_train, * from s_35926_dataset_alipay_valid
        union all
        select 1 is_train, * from s_35926_dataset_alipay_train
        ''')
