
### 三个实验的参数

* alimama 数据集

```python
    batch_size = 1024
    params = dict(
        learning_rate=0.001,
        optimizer='adam',
        l2_reg=1e-5, l2_reg_embedding=1e-5, reg_type=3,
        embedding_size=4, hist_embedding_size=8,
        dnn_hidden_units=(128, 256, 80, 256),
        pcoc=0.01,  # 0.0
        featmap=dict(
                sp_feats={'userid': 1141729+1,
                          'adgroup_id': 846811+1,
                          'pid': 2,
                          'cms_segid': 97,
                          'cms_group_id': 13,
                          'final_gender_code': 2,
                          'age_level': 7,
                          'pvalue_level': 4,
                          'shopping_level': 3,
                          'occupation': 2,
                          'new_user_class_level': 5,
                          'campaign_id': 423436+1,
                          'customer': 255875+1,
                          },
                query_feats={'cate_id': 12977 + 1, 'brand': 461528 + 1},
                hist_feats={'hist_cate_id': 12977 + 1, 'hist_brand': 461528 + 1},
                dense_feat=['price'],
                ),
        value_name={'hist_brand': ['count', 'time'], 'hist_cate_id': ['count', 'time']},
        boundaries_time=[10 * 60,  60 * 60, 3*3600, 6*3600, 86400, 2*86400, 4*86400, 7*86400, 14*86400],
        boundaries_count=[0.5 + i for i in range(10)],
        is_item_wise=1,
        is_din_2=1,
        )
```



*  taobao 数据集

```python
    batch_size = 128
    params = dict(
        learning_rate=0.001,
        optimizer='adam',
        l2_reg=1e-5, l2_reg_embedding=1e-5, reg_type=3,
        embedding_size=16, hist_embedding_size=16,
        dnn_hidden_units=(128, 256, 80, 256),
        pcoc=0.01,  # 0.0
        featmap={'sp_feats': None, 'query_feats': {'item': 4162024, 'cate': 9439},
                 'hist_feats': {'hist_item': 4162024, 'hist_cate': 9439}},
        value_name={'hist_item': ['cate', 'time'], 'hist_cate': ['time']},
        boundaries_time=[86400/2, 86400, 2*86400, 3*86400, 4*86400, 5*86400, 6*86400],
        n_cate=9439,
        is_item_wise=1,
        is_din_2=1,
        )
```




* alipay 数据集

```python
    batch_size = 100
    params = dict(
        learning_rate=0.001,
        optimizer='adam',
        l2_reg=1e-4, l2_reg_embedding=1e-4, reg_type=3,
        embedding_size=16, hist_embedding_size=16,
        dnn_hidden_units=(200, 80),
        pcoc=0.01,  # 0.0
        featmap={'sp_feats': None, 'query_feats': {'item': 2200291, 'cate': 72, 'sid': 9999},
                 'hist_feats': {'hist_item': 2200291, 'hist_cate': 72, 'hist_sid': 9999}},
        value_name={'hist_item': ['cate', 'count', 'time'], 'hist_cate': ['count', 'time'],
                    'hist_sid': ['count', 'time']},
        boundaries_time=[86400, 2*86400, 4*86400, 7*86400, 15*86400, 30*86400, 90*86400],
        boundaries_count=[0.5 + i for i in range(10)],
        n_cate=72,
        is_item_wise=0,
        is_din_2=1,
        keep_prob=0.8,
        bn=True, mode='softmax', item_initializer=1,
        )
```