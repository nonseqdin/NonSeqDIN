# coding=utf-8
import tensorflow as tf
import pprint


def build_graph(features, y=None, **kwargs):
    '''
    features: 字典, 格式为, 特征名称: Tensor
    y: tensor tf.int32
    '''
    tf.logging.info('build_graph:params: %s' % pprint.pformat(kwargs))
    from model_dinmp import DINMP
    keep_prob = kwargs.pop('keep_prob', 1)
    logits = DINMP(features,
                   featmap=kwargs.pop('featmap'),
                   embedding_size=kwargs.pop('embedding_size', 4),
                   hist_embedding_size=kwargs.pop('hist_embedding_size', 4),
                   l2_reg_embedding=kwargs.pop('l2_reg_embedding', 1e-5),
                   l2_reg=kwargs.pop('l2_reg', 1e-5),
                   dnn_hidden_units=kwargs.pop('dnn_hidden_units', (128, 256, 256)),
                   reg_type=kwargs.pop('reg_type', 1),
                   interaction_order=kwargs.pop('interaction_order', 1),
                   keep_prob=1 if y is None else keep_prob,
                   **kwargs,
                   )

    # output
    softmax_score = tf.nn.softmax(logits)
    prediction_result = tf.argmax(softmax_score, 1)
    pred_score = tf.reshape(softmax_score[:, 1], [-1, 1])
    predict_prob = pred_score

    # ## 调试神器!!
    loss, metrics, others, hooks = None, None, None, None
    if kwargs.get("debug", False):
        hooks = [tf.train.LoggingTensorHook({"debug: predict_prob = ": predict_prob}, every_n_iter=10)]
    if y is not None:
        pcoc = tf.abs(tf.reduce_mean(pred_score)/tf.reduce_mean(tf.cast(y, tf.float32)) - 1,
                      name='pcoc') * max(0.0, kwargs.get('pcoc', 0.0))
        label = tf.one_hot(tf.reshape(y, [-1]), 2)
        loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))
        loss = loss0 + tf.losses.get_regularization_loss() + pcoc
        loss1 = loss - loss0
        metrics = {'accuracy': tf.metrics.accuracy(labels=y, predictions=prediction_result, name='acc_op'),
                   'auc': tf.metrics.auc(y, pred_score, curve="ROC", name='auc_op',  num_thresholds=2000)}
        others = {'loss0': loss0, 'regloss': loss1, 'loss_pcoc': pcoc, 'debug_hooks': hooks}
    return predict_prob, loss, metrics, others


class BaseClassifier(tf.estimator.Estimator):

    def __init__(self, tf_config, params):

        def _model_fn(features, labels, mode, params):

            def get_labels():
                _labels = list(labels.values())
                return _labels if len(_labels) > 1 else _labels[0]

            y = None if mode == tf.estimator.ModeKeys.PREDICT else get_labels()
            # 构造图
            predict_prob, loss, metrics, others = build_graph(features, y, **params)
            # others: a dict: debug_hooks, loss0,loss1
            debug_hooks = None if others is None else others.pop('debug_hooks', None)

            if mode == tf.estimator.ModeKeys.PREDICT:   # 预测
                output = {'prediction': tf.estimator.export.PredictOutput({'sigmoid': predict_prob})}
                return tf.estimator.EstimatorSpec(
                        mode,
                        predictions=predict_prob,
                        export_outputs=output,
                        )
            if mode == tf.estimator.ModeKeys.TRAIN:  # 训练
                tf.summary.scalar('train_loss', loss)
                tf.summary.scalar('train_auc', metrics['auc'][0])
                summary_op = tf.summary.merge_all()
                summary_hook = tf.train.SummarySaverHook(
                        save_steps=100,
                        output_dir=tf_config.model_dir,
                        summary_op=summary_op,
                        )
                learning_rate = params.get('learning_rate', 0.0001)
                optimizer_name = params.get('optimizer', 'group_adam')
                assert optimizer_name in ('adam', 'adagrad'), '选择支持的优化器'
                if optimizer_name == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                elif optimizer_name == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate)
                train = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(
                        mode,
                        loss=loss,
                        train_op=train,
                        training_hooks=[summary_hook],
                        predictions=predict_prob,
                        training_chief_hooks=debug_hooks  # [model_export_hook]
                        )
            if mode == tf.estimator.ModeKeys.EVAL:  # 验证
                return tf.estimator.EstimatorSpec(
                        mode,
                        loss=loss,
                        eval_metric_ops=metrics,
                        # evaluation_hooks=hooks
                        )

        super(BaseClassifier, self).__init__(
                model_fn=_model_fn,
                model_dir=tf_config.model_dir,
                config=tf_config,
                params=params
                )
