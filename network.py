import tensorflow as tf 




def value_model(features, y, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    y_hat = tf.layers.dense(net, 1, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': y_hat,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)

    # Compute evaluation metrics.
    mean_sq_error = tf.metrics.mean_squared_error(labels=y,
                                   predictions=y_hat,
                                   name='error_op')
    metrics = {'mean_squared_error': mean_sq_error}
    tf.summary.scalar('accuracy', mean_sq_error[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class ValueF_net()
    #This class is used to predict the value function
    def __init__( ):



    def _net_init(feature_columns):

    	classifier = tf.estimator.Estimator(
        model_fn=value_model,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [100, 100],
            'n_classes': 3,
        })



class Policy_net










