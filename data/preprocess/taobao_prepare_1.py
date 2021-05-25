# import cPickle as pkl
import pickle
import pandas as pd
import random

RAW_DATA_FILE = './data/taobao_data/UserBehavior.csv'
DATASET_PKL = './data/taobao_data/dataset.pkl'


def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df


def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))

    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print(f'item_len={item_len}, user_len={user_len}, cate_len={cate_len}, btag_len={btag_len}')
    return df, item_len, user_len + item_len + cate_len + btag_len + 1  # +1 is for unknown target btag


def gen_user_item_group(df, item_cnt, feature_size):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, feature_size):
    train_sample_list = []
    test_sample_list = []

    # get each user's last touch point time

    print(len(user_df))

    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        if cnt % 10000 == 1:
            print('cnt = ', cnt)
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        # btag_hist = hist['btag'].tolist()

        target_item_time = hist['time'].tolist()[-1]
        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        # target_item_btag = feature_size

        label = 1
        is_train = 1 - (target_item_time > split_time)

        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
                # target_item_btag = feature_size

        tt = (hist['time'].values[-1] - hist['time'].values[:-1]).tolist()
        hist_item = '|'.join('{}:{}^{}'.format(k, v1, v2) for k, v1, v2 in zip(item_hist[:-1], cate_hist[:-1], tt))
        hist_cate = '|'.join('{}:{}'.format(k, v1) for k, v1 in zip(cate_hist[:-1], tt))

        if is_train == 1:
            train_sample_list.append((uid, is_train, target_item, target_item_cate, hist_item, hist_cate, label))
        else:
            test_sample_list.append((uid, is_train, target_item, target_item_cate, hist_item, hist_cate, label))

    train_sample_length_quant = len(train_sample_list) // 256 * 256
    test_sample_length_quant = len(test_sample_list) // 256 * 256

    print("length", len(train_sample_list), train_sample_length_quant)
    train_sample_list = train_sample_list[:train_sample_length_quant]
    test_sample_list = test_sample_list[:test_sample_length_quant]
    random.shuffle(train_sample_list)
    print("length", len(train_sample_list), 'test', len(test_sample_list))
    return train_sample_list, test_sample_list


def main():
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df)
    print("feature_size", item_cnt, feature_size)
    feature_total_num = feature_size + 1

    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)
    train_sample_list, test_sample_list = gen_dataset(user_df, item_df, item_cnt, feature_size)

    with open(DATASET_PKL, 'wb') as f:
        pickle.dump(train_sample_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_sample_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feature_total_num, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()


'''
df = to_df(RAW_DATA_FILE)
df, item_cnt, feature_size = remap(df)
print("feature_size", item_cnt, feature_size)

# #item = 4162024  #user = 987994 #cate = 9439  4
# feature_size #item = 4162024 #feature = 5159462
# length 691456  / 256 = 2701,  test 296192
'''
