import tensorflow as tf
from tensorflow.contrib.layers import sparse_column_with_integerized_feature, embedding_column


def get_embedding(features, sp_feats, embedding_size, l2_reg=0):
    from common import sp_cast
    feature_columns = []
    for feature in sp_feats:
        col = sparse_column_with_integerized_feature(feature, sp_feats[feature])
        embedding_col = embedding_column(sparse_id_column=col, dimension=embedding_size,
                                         initializer=tf.contrib.layers.variance_scaling_initializer(seed=2020))
        feature_columns.append(embedding_col)
    emb = tf.contrib.layers.input_from_feature_columns(
            columns_to_tensors={k: sp_cast(features[k]) for k in sp_feats},  # features
            feature_columns=feature_columns)

    if l2_reg > 0:
        name = '_'.join(sp_feats.keys())[:10]
        t_l2_reg = tf.reduce_mean(tf.reduce_sum(emb**2, axis=-1), name=name) * l2_reg
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, t_l2_reg)
    return emb


def DINMP(features, featmap=None, hist_embedding_size=20, embedding_size=4,
          dnn_hidden_units=(128, 256, 256), l2_reg_embedding=1e-6, l2_reg=0, reg_type=1,
          **kwargs):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_hidden_units: list of positive integer or empty, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector

    :return: logit
    """
    sp_feats = featmap.get('sp_feats')
    dense_feats = featmap.get('dense_feat')
    query_feats = featmap['query_feats']
    hist_feats = featmap['hist_feats']

    with tf.variable_scope('embedding_attention_scope', reuse=tf.AUTO_REUSE):
        from common import get_interest_network
        hist1, query_emb = get_interest_network(
                features, query_feats, hist_feats,
                embedding_size=hist_embedding_size, l2=l2_reg_embedding, reg_type=reg_type, **kwargs)

    _emb_list = [hist1, query_emb]
    if dense_feats is not None:
        dense_value = tf.concat([features[k] for k in dense_feats], -1)
        _emb_list.append(dense_value)
    if sp_feats is not None:
        sparse_emb = get_embedding(features, sp_feats, embedding_size, l2_reg=l2_reg)
        _emb_list.append(sparse_emb)

    merged = tf.concat(_emb_list, -1)

    for u in dnn_hidden_units:
        merged = tf.layers.dense(inputs=merged, units=u, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=merged, units=2)
    return logits
