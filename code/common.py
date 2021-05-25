import collections
import tensorflow as tf
from tensorflow.contrib.layers.python.ops import bucketization_op
from tf_string_utils import string_kv_to_sparse


def DEBUG(name, Tensor):
    # from penrose.util import context_util
    # context_instance = context_util.get_instance()
    # context_instance.add_to_log_collection(name, Tensor)
    pass


def sp_cast(sp, out_dtype=tf.int32):
    if sp.dtype == tf.string:
        return tf.SparseTensor(sp.indices, tf.string_to_number(sp.values, out_dtype), sp.dense_shape)
    return tf.cast(sp, out_dtype)


def embedding_lookup_unique(E, J):
    unique_ids, idx = tf.unique(J)
    unique_embeddings = tf.nn.embedding_lookup(E, unique_ids)
    embeds_flat = tf.gather(unique_embeddings, idx)
    return embeds_flat


def minibatch_reg(weight_matrix, J, n_rows, l1=0, l2=0, reg_type=1):
    I_unique, _, I_count = tf.unique_with_counts(J)
    vals = tf.gather(weight_matrix, I_unique)

    def _get_reg(typo, lambdo):
        if lambdo <= 0:
            return 0
        _m = {'l2': tf.square(vals), 'l1': tf.abs(vals)}
        numerator = tf.reduce_sum(_m[typo], axis=1)
        if reg_type == 1:
            denominator = tf.sqrt(tf.cast(I_count, tf.float32))
        elif reg_type == 2:
            denominator = n_rows / tf.cast(I_count, tf.float32)
        else:
            denominator = tf.sqrt(n_rows / tf.cast(I_count, tf.float32))
        return tf.reduce_sum(numerator / denominator)

    return l2 * _get_reg('l2', l2) + l1 * _get_reg('l1', l1)


def din_attention(queries, keys, mode=1):
    '''
    queries:     [B, H1]
    keys:        [B, T, H]
    '''
    H = keys.shape.as_list()[-1]
    T = tf.shape(keys)[1]
    queries = tf.layers.dense(inputs=queries, units=H, use_bias=False)
    if mode == 0:
        queries = tf.tile(queries, [1, T])
        queries = tf.reshape(queries, [-1, T, H])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, units=80,
                                        activation=tf.nn.sigmoid)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, units=40,
                                        activation=tf.nn.sigmoid)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, units=1)
        outputs = d_layer_3_all  # [B, T, 1]
        # Scale
        outputs = outputs / (H**0.5)
        # Activation
        outputs = tf.nn.softmax(outputs, axis=1)  # [B, T, 1]
        # Weighted sum
        outputs = tf.reshape(outputs, [-1, 1, T])
        outputs = tf.matmul(outputs, keys)        # [B, 1, H]
    else:
        outputs = keys @ tf.expand_dims(queries, -1)  # [B, T, 1]
        outputs = tf.sigmoid(outputs)
        outputs = outputs * keys
    return outputs


# ############################
# 非序列处理方法的主要的逻辑
# ############################

SparseInput = collections.namedtuple('SparseKeyVector', ['indices', 'value_list', 'value_name', 'dense_shape'])


def _get_interest_network(sp_input, feat_name, E, query_emb, max_id, embedding_size, suffix='', **kwargs):
    '''
    **kwargs:
       boundaries_count, boundaries_time, n_cate
       is_item_wise
       model: sdin, edin
       partition_cate
       is_din_2
    '''
    I_r, J_c = sp_input.indices[:, 0], sp_input.indices[:, 1]
    dim = kwargs.get('dim', 3)

    def target_attention_sore(query_emb, E_J, mode='sigmoid'):
        # score = sigmoid(<query_emb, key_emb>)
        # ## 可优化 ##
        Q_I = tf.gather(query_emb, I_r)
        Q_I = tf.layers.dense(inputs=Q_I, units=E_J.shape.as_list()[-1], use_bias=False)
        # ###########
        if mode == 'softmax':
            vv = tf.exp(tf.reduce_sum(Q_I * E_J, axis=1))
            sum_vv = tf.segment_sum(vv, I_r)
            return vv / tf.gather(sum_vv, I_r)
        # mode == 'sigmoid':
        return tf.sigmoid(tf.reduce_sum(Q_I * E_J, axis=1))

    def val_to_param(val, n_val, name, u1_J):
        # map the value to a parameter for each item, or for all items.
        if kwargs.get('is_item_wise', 1) and n_val * max_id < 1e8:  # 避免内存不足, bug:原代码1e-8,可能影响mama
            u2 = tf.get_variable(name, initializer=tf.constant(1.0/dim, shape=[n_val, dim]))
            mU = tf.reduce_sum(u1_J * tf.gather(u2, val), -1)
        else:
            u2 = tf.get_variable(name, initializer=tf.constant(1.0, shape=[n_val]))
            mU = tf.gather(u2, val)
        return mU

    def embedding_group(weight_emb, J_by, n_rows, n_cols):
        Idx = I_r * n_cols + J_by
        emb_group = tf.unsorted_segment_sum(weight_emb, Idx,  n_cols * n_rows)
        emb_group = tf.reshape(emb_group, [n_rows, n_cols, embedding_size])
        return emb_group

    # 1. 处理DIN的attention sore
    E_J = embedding_lookup_unique(E, J_c)
    M1 = target_attention_sore(query_emb, E_J, kwargs.get('mode'))
    if kwargs.get('model') == 'sdin':
        weight_emb = tf.expand_dims(M1, 1) * E_J
        emb = tf.unsorted_segment_sum(weight_emb, I_r, sp_input.dense_shape[0])
        return emb

    u1_J = None  # 用于影响因子的个性化计算
    if kwargs.get('is_item_wise', 1):
        u1 = tf.get_variable(f'{feat_name}_u1_{suffix}', initializer=tf.constant(1.0, shape=[max_id, dim]))
        u1_J = tf.gather(u1, J_c)
    I_buckets, V_buckets, T_buckets, N_buckets = [], [], [], []
    for i, v_name in enumerate(sp_input.value_name):
        if v_name not in ('cate', 'time', 'count'):
            continue
        if v_name == 'cate':
            # 2. 处理 cate
            n_cols = kwargs['n_cate']
            bi = tf.cast(sp_input.value_list[i], tf.int64)
        else:   # v_name in ('time', 'count'):
            # 3. 处理 time, count
            bd = kwargs[f'boundaries_{v_name}']
            n_cols = len(bd) + 1
            bi = tf.cast(bucketization_op.bucketize(sp_input.value_list[i], bd), tf.int64)
        I_buckets.append(bi)
        V_buckets.append(val_to_param(bi, n_cols, f'{feat_name}_{i}_{suffix}', u1_J))
        if v_name in ('time', 'count'):
            u2 = tf.get_variable(f'{feat_name}_{i}_{suffix}')
            DEBUG(f'{feat_name}_{i}_{suffix}_{v_name}', u2)
        T_buckets.append(v_name)
        N_buckets.append(n_cols)

    # 4. nonlinear layer
    n_val = len(T_buckets)
    if n_val > 1:
        o2 = tf.reduce_prod(tf.stack(V_buckets, 0), 0)
        M2 = tf.stack(V_buckets + [o2, M1 * o2, M1], axis=1)
    else:
        M2 = tf.stack([V_buckets[0], M1 * V_buckets[0], M1], axis=1)
    # 5. output new score
    M = tf.layers.dense(M2, 1)
    weight_emb = M * E_J

    if kwargs.get('model') == 'edin':
        emb = tf.unsorted_segment_sum(weight_emb, I_r, sp_input.dense_shape[0])
        return emb

    # ## 新增模型代码分组
    print('_get_interest_network::new_model')
    emb_list, n1 = [], 0
    for I_b, v_name, n_cols in zip(I_buckets, T_buckets, N_buckets):
        if v_name == 'time' \
                or (v_name == 'cate' and kwargs.get('partition_cate')):
            # # 汇总 (batch_size, n_cols, embedding_size)
            embs_p = embedding_group(weight_emb, I_b, n_rows=sp_input.dense_shape[0], n_cols=n_cols)
            if v_name == 'cate' and kwargs.get('partition_cate') and n_cols > 100:
                weight_30 = tf.get_variable(
                        f'{v_name}_to_30',
                        shape=[n_cols, 30],
                        initializer=tf.contrib.layers.variance_scaling_initializer(seed=2020)
                        )
                n_cols = 30
                embs_p = tf.einsum('ijk,jl->ilk', embs_p, weight_30)
            emb_list.append(embs_p)
            n1 += n_cols
    # # 汇总的    (batch_size, 1, embedding_size)
    embs_1 = tf.unsorted_segment_sum(weight_emb, I_r, sp_input.dense_shape[0])
    embs_1 = tf.expand_dims(embs_1, 1)
    emb_list.append(embs_1)
    n1 += 1
    # 合并        (batch_size, n1, embedding_size)
    embs = tf.concat(emb_list, axis=1)
    # din 处理
    if kwargs.get('is_din_2') == 1:
        embs = din_attention(query_emb, embs)
    emb = tf.reshape(embs, shape=(-1, n1 * embedding_size))
    return emb


def get_interest_network(features, query_feats, hist_feats,
                         embedding_size=20, l2=0, l1=0, reg_type=1, **kwargs):
    emb_inits = {0: tf.contrib.layers.variance_scaling_initializer(seed=2020), 1: tf.truncated_normal_initializer}
    # 读入数据, 并定义embedding矩阵
    emb_map, input_map = {}, {}
    for name, max_id in hist_feats.items():
        value_name = kwargs['value_name'][name]
        indices, value_list, dense_shape = string_kv_to_sparse(
                features[name], max_id, num_vals=len(value_name), is_list=True)
        input_map[name] = SparseInput(indices, value_list, value_name, dense_shape)

        feat_name = name.split('hist_')[-1]
        E = tf.get_variable(f"{feat_name}_emb_weights",
                            initializer=emb_inits[kwargs.get(f'{feat_name}_initializer', 0)],
                            regularizer=lambda w: minibatch_reg(w, indices[:, 1], max_id, l1, l2, reg_type
                                                                ) if l2 + l1 > 0 else None,
                            shape=[max_id, embedding_size])
        emb_map[name] = E

    # 取query_feats的embeddings
    query_embs_map = {}
    for feat_name, max_id in query_feats.items():
        E = tf.get_variable(f"{feat_name}_emb_weights",
                            initializer=emb_inits[kwargs.get(f'{feat_name}_initializer', 0)],
                            shape=[max_id, embedding_size])
        sp_x = sp_cast(features[feat_name], tf.int64)
        query_embs_map[feat_name] = tf.nn.safe_embedding_lookup_sparse(E, sp_x)

    # query_emb 计算逻辑
    query_emb = tf.concat(list(query_embs_map.values()), axis=-1)

    if l2 > 0:
        t_l2_reg = tf.reduce_mean(tf.reduce_sum(query_emb**2, axis=-1), name='query_emb') * l2
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, t_l2_reg)

    # 核心的计算逻辑
    emb_list = []
    for name, max_id in hist_feats.items():
        emb = _get_interest_network(input_map[name], name, emb_map[name], query_emb,
                                    max_id, embedding_size, **kwargs)
        emb_list.append(emb)
    emb = tf.concat(emb_list, axis=-1)
    return emb, query_emb
