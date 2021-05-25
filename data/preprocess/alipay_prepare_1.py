import pickle
import pandas as pd
import random
random.seed(1111)

RAW_DATA_FILE = './data/alipay_data/ijcai2016_taobao.csv'
DATASET_PKL = './data/alipay_data/dataset.pkl'


def to_df(file_name=RAW_DATA_FILE):
    df = pd.read_csv(file_name)
    df.columns = ['uid', 'sid', 'iid', 'cid', 'btype', 'time']  # use_ID,sel_ID,ite_ID,cat_ID,act_ID,time
    print(f'all df shape = {df.shape}')
    df = df.query("btype == 0").copy()
    print(f'filtered df shape = {df.shape}')

    # 日期转化为秒, 算时间差
    df.time = pd.to_datetime(df.time, format='%Y%m%d').astype(int) // int(1e9)
    # import time
    # from datetime import datetime
    # df.time = df.time.map(lambda dt: int(time.mktime(datetime.strptime(str(dt), "%Y%m%d").timetuple())))
    return df


def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))

    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    seller_key = sorted(df['sid'].unique().tolist())
    seller_len = len(seller_key)
    seller_map = dict(zip(seller_key, range(seller_len)))
    df['sid'] = df['sid'].map(lambda x: seller_map[x])

    print(f'item_len={item_len}, user_len={user_len}, cate_len={cate_len}, seller_len={seller_len}')
    return df, item_len, user_len + item_len + cate_len + seller_len


def gen_user_item_group(df):

    aa = df.uid.value_counts()
    uids = aa.index[aa > 3]
    user_df = df[df.uid.isin(uids)].sort_values(['uid', 'time']).groupby('uid')

    item_df = df[['iid', 'cid', 'sid']].drop_duplicates()
    item_df.index = item_df.iid

    n1, n2 = len(aa), len(uids)
    print(f"#users {n1} /{n2} group completed")
    return user_df, item_df


def neg_sample(user_seq):  # 注意这部分逻辑与对比论文中保持一致
    r = random.randint(0, 4)
    if r == 0:
        return random.randint(0, 2200290)  # 从iid中随机选
    else:
        return random.choice(user_seq)


def gen_dataset(user_df, item_df, item_cnt):
    train_sample_list = []
    valid_sample_list = []
    test_sample_list = []

    cnt = 0
    for uid, _hist in user_df:
        cnt += 1
        if cnt % 10000 == 1:
            print('cnt = ', cnt)

        #  hist: uid   sid  iid  cid  btype   time
        hist = _hist[-400:] if _hist.shape[0] > 400 else _hist

        def create_hist(flag):
            assert flag in (-3, -2, -1), 'train: -3, valid: -2, test: -1'
            tt = (hist['time'].values[flag] - hist['time'][:-3]).tolist()

            in_zip = zip(hist['iid'][:-3], hist['cid'][:-3], hist['sid'][:-3], tt)
            hist_item = '|'.join('{}:{}^{}^{}'.format(k, v1, v2, t) for k, v1, v2, t in in_zip)
            hist_cate = '|'.join('{}:{}'.format(k, v1) for k, v1 in zip(hist['cid'][:-3], tt))
            hist_seller = '|'.join('{}:{}'.format(k, v1) for k, v1 in zip(hist['sid'][:-3], tt))

            return hist_item, hist_cate, hist_seller

        label1, label0 = 1, 0

        hist_item, hist_cate, hist_seller = create_hist(-3)
        iid, cid, sid = item_df.loc[_hist['iid'].values[-3]]
        train_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label1])
        iid, cid, sid = item_df.loc[neg_sample(_hist['iid'].values[:-3])]
        train_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label0])

        hist_item, hist_cate, hist_seller = create_hist(-2)
        iid, cid, sid = item_df.loc[_hist['iid'].values[-2]]
        valid_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label1])
        iid, cid, sid = item_df.loc[neg_sample(_hist['iid'].values[:-3])]
        valid_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label0])

        hist_item, hist_cate, hist_seller = create_hist(-1)
        iid, cid, sid = item_df.loc[_hist['iid'].values[-1]]
        test_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label1])
        iid, cid, sid = item_df.loc[neg_sample(_hist['iid'].values[:-3])]
        test_sample_list.append([uid, iid, cid, sid, hist_item, hist_cate, hist_seller, label0])

    # random.shuffle(train_sample_list)
    print("length", len(train_sample_list), 'valid', len(valid_sample_list), 'test', len(test_sample_list))
    return train_sample_list, valid_sample_list, test_sample_list


def main():
    df = to_df()
    df, item_cnt, feature_size = remap(df)
    print("feature_size", item_cnt, feature_size)

    user_df, item_df = gen_user_item_group(df)
    train_sample_list, valid_sample_list, test_sample_list = gen_dataset(user_df, item_df, item_cnt)

    with open(DATASET_PKL, 'wb') as f:
        pickle.dump(train_sample_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_sample_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_sample_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

'''
all df shape = (44528127, 6)
filtered df shape = (35179371, 6)
item_len=2200291, user_len=626041, cate_len=72, seller_len=9999
feature_size 2200291 2836403
#users 626041 /498308 group completed
length 996616 valid 996616 test 996616
'''
