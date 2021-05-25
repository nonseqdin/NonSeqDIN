import tensorflow as tf


def string_seq_to_tensor(lines, max_len=64, sep=',', out_type=tf.float32, mode=1):
    '''
    lines = ['1,2,3', '3,4']
    返回：
    array([[1., 2., 3.],
          [3., 4., 0.]], dtype=float32)
    说明: 同PAITF的函数trans_csv_id2dense(records, max_id, id_as_value=True)
    '''
    sps = tf.string_split(lines, delimiter=sep)
    spss = tf.sparse_slice(sps, [0, 0], [sps.dense_shape[0], int(max_len)])
    dense_shape = tf.stack((spss.dense_shape[0], int(max_len)))
    sp_x = tf.SparseTensor(indices=spss.indices,
                           values=tf.string_to_number(spss.values, out_type=out_type),
                           dense_shape=spss.dense_shape if mode == 1 else dense_shape)
    return tf.sparse.to_dense(sp_x)


def string_kv_to_tensor(lines, max_len=64, sep='|', kv_sep=':', val_sep='^', num_vals=3, out_type=tf.float32):
    '''
    lines = ["12:4^3^3|1:5^3^3|88:6^3^3|1:3^3^3|2:100^3^3", "12:4^3^3|1:5^3^3"]
    key, v1, v2, v3 = string_kv_to_tensor(lines, 100)
    '''
    sps = tf.string_split(lines, delimiter=sep)
    spss = tf.sparse_slice(sps, [0, 0], [sps.dense_shape[0], int(max_len)])

    splits = tf.string_split(spss.values, kv_sep)
    id_vals = tf.reshape(splits.values, splits.dense_shape)
    col_ids, vals = tf.split(id_vals, num_or_size_splits=2, axis=1)

    def to_dense(vs):
        _vs = tf.string_to_number(vs, out_type=out_type)
        _vs = tf.SparseTensor(indices=spss.indices, values=_vs,
                              dense_shape=spss.dense_shape)
        return tf.sparse.to_dense(_vs)

    vv = []
    # key
    vv.append(to_dense(col_ids[:, 0]))
    # values
    if num_vals > 1:
        _vals = tf.string_split(vals[:, 0], val_sep)
        _vals = tf.reshape(_vals.values, _vals.dense_shape)
        _vals = tf.split(_vals, num_or_size_splits=num_vals, axis=1)
        _list = [to_dense(v[:, 0]) for v in _vals]
        vv.extend(_list)
    if num_vals == 1:
        vv.append(to_dense(vals[:, 0]))
    return vv


def string_kv_to_sparse(lines, num_cols, sep='|', kv_sep=':', val_sep='^', num_vals=3, hash_key=False, is_list=False):
    '''
    lines = ["12:4^3^3|1:5^3^3|88:6^3^3|1:3^3^3|2:100^3^3", "12:4^3^3|1:5^3^3"]
    indices, (v1, v2, v3), dense_shape = string_kv_to_sparse(lines, 100)
    '''
    if isinstance(lines, tf.SparseTensor):
        columns = tf.string_split(lines.values, sep)
        num_rows = lines.dense_shape[0]
    else:
        columns = tf.string_split(lines, sep)
        num_rows = columns.dense_shape[0]

    splits = tf.string_split(columns.values, kv_sep)
    id_vals = tf.reshape(splits.values, splits.dense_shape)
    col_ids, vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    if hash_key:
        col_ids = tf.string_to_hash_bucket_fast(col_ids[:, 0], num_cols)
    else:
        col_ids = tf.string_to_number(col_ids[:, 0], out_type=tf.int64)

    indices = tf.stack((columns.indices[:, 0], col_ids), axis=-1)
    dense_shape = tf.stack([num_rows, num_cols])

    if num_vals > 1:
        _vals = tf.string_split(vals[:, 0], val_sep)
        _vals = tf.reshape(_vals.values, _vals.dense_shape)
        _vals = tf.split(_vals, num_or_size_splits=num_vals, axis=1)
        values_list = [tf.string_to_number(v[:, 0], out_type=tf.int64) for v in _vals]
        return indices, values_list, dense_shape
    if num_vals == 1:
        values = tf.string_to_number(vals[:, 0], out_type=tf.float32)
        if is_list:
            values = [values]
        return indices, values, dense_shape
    return None
