import copy




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def path2feat(path):
    raise NotImplementedError


def update_cfg(def_cfg, usr_cfg):
    #input: dict of default cfg and dict of usr cfg, but not change the original one
    if usr_cfg is None:
        return def_cfg
    out = copy.deepcopy(def_cfg)
    out.update(usr_cfg)
    return out 



def var_shape(x):
    #calculate the shape of a tensor and return a list 
    return [k.value for k in x.get_shape()]


def num_ele(x):
    #calculate the number of elements in a tensor
    return np.prod(var_shape(x))

#compute grad and stretch it to 1-dimension
def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [num_ele(v)])
                         for (v, grad) in zip(var_list, grads)])



class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

        #change variable in the var_list to value of corresponding value in theta.    theta -> var_list
    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})



class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self, a, prob):
        raise NotImplementedError
    def loglikelihood(self, a, prob):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.ivector('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return prob[T.arange(prob.shape[0]), a]
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        return distributions.categorical_sample(prob)
    def maxprob(self, prob):
        return prob.argmax(axis=1)