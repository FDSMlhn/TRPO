import tensorflow as tf
import six

class OnlineEstimator(tf.estimator.Estimator):

    def __init__(self, *args, **kwargs):
        


    def online_predict(self,
                input_fn,
                predict_keys=None,
                hooks=None):
        """Yields predictions for given features using online model rather than checkpoint model.

        Args:
          input_fn: Input function returning features which is a dictionary of
            string feature name to `Tensor` or `SparseTensor`. If it returns a
            tuple, first item is extracted as features. Prediction continues until
            `input_fn` raises an end-of-input exception (`OutOfRangeError` or
            `StopIteration`).
          predict_keys: list of `str`, name of the keys to predict. It is used if
            the `EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used
            then rest of the predictions will be filtered from the dictionary. If
            `None`, returns all.
          hooks: List of `SessionRunHook` subclass instances. Used for callbacks
            inside the prediction call.

        Yields:
          Evaluated values of `predictions` tensors.

        Raises:
          ValueError: Could not find a trained model in model_dir.
          ValueError: if batch length of predictions are not same.
          ValueError: If there is a conflict between `predict_keys` and
            `predictions`. For example if `predict_keys` is not `None` but
            `EstimatorSpec.predictions` is not a `dict`.
        """
        #hooks = tf.estimator._check_hooks_type(hooks)
        # Check that model has been trained.

        with tf.Graph().as_default() as g, g.device(self._device_fn):
            tf.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features = self._get_features_from_input_fn(
                input_fn, tf.estimator.ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(
                features, None, tf.estimator.ModeKeys.PREDICT, self.config)
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            with tf.train.MonitoredSession(
                    session_creator=tf.train.WorkerSessionCreator(
                        master=self._config.master,
                        scaffold=estimator_spec.scaffold,
                        config=self._session_config),
                    hooks=hooks) as mon_sess:
                while not mon_sess.should_stop():
                    preds_evaluated = mon_sess.run(predictions)
                    if not isinstance(predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(self._extract_batch_length(preds_evaluated)):
                            yield {
                                key: value[i]
                                for key, value in six.iteritems(preds_evaluated)
                            }
