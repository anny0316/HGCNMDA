from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class EdgeDecoder(Layer):
    def __init__(self, idx, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(EdgeDecoder, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_diease_nonzero = placeholders['num_features_diease_nonzero']
        self.features_g = placeholders['features_g']
        self.adj_m2g = placeholders['adj_m2g']
        self.adj_d2g = placeholders['adj_d2g']
        self.idx = idx

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
            # for i in range((self.d2r_adj).shape[1]):
                self.vars['relation_' + str(i)] = glorot([input_dim, output_dim], 0.001,
                                                        name='relation_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        supports = list()
#         inputs_row = tf.nn.dropout(inputs, 1 - self.dropout)
#         inputs_col = tf.nn.dropout(inputs, 1 - self.dropout)
#         edge = tf.gather(inputs, self.train_edge)
#         edge = tf.reshape(edge,[self.train_edge[0],-1])
#         inputs_row = tf.gather(inputs_row,(self.train_edge)[:,0])
#         inputs_col = tf.gather(inputs_col, (self.train_edge)[:,1])
        # dropout
        # if self.sparse_inputs:
        #     inputs_bak = inputs
        #     inputs = tf.sparse_tensor_to_dense(inputs)
        #     inputs = tf.reshape(inputs,(1,-1))
        #     mask = tf.greater(inputs,0.0)
        #     non_zero_array = tf.boolean_mask(inputs, mask)
        #     sha = tf.shape(non_zero_array)
        #     inputs = sparse_dropout(inputs_bak, 1 - self.dropout, sha )  #tf.count_nonzero(inputs)
        # else:
        #     inputs = tf.nn.dropout(inputs, 1 - self.dropout)

        for i in range(len(self.support)):
            if not self.featureless:
                intermediate_product = dot(inputs, self.vars['relation_' + str(i)], sparse=self.sparse_inputs)

            if self.idx==0:
                rec = dot(self.adj_m2g, intermediate_product, sparse=False)
            else:
                rec = dot(self.adj_d2g, intermediate_product, sparse=False)
            supports.append(rec)
        output = tf.add_n(supports)
        # output = rec
        output = tf.nn.l2_normalize(output, dim=1)
        return self.act(output)

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], 0.001,
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # output = support

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class d2m_EdgeDecoder(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(d2m_EdgeDecoder, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_diease_nonzero = placeholders['num_features_diease_nonzero']
        self.mirna_length = placeholders['mirna_length']
        self.diease_length = placeholders['diease_length']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['d2m_relation_' + str(i)] = glorot([input_dim, output_dim], 0.001,
                                                        name='d2m_relation_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        supports = list()

        d2g = tf.slice(inputs,[0,0],[self.diease_length,-1])
        m2g = tf.slice(inputs,[self.diease_length,0],[self.mirna_length,-1])
        for i in range(len(self.support)):
            if not self.featureless:
                intermediate_product = dot(d2g, self.vars['d2m_relation_' + str(i)], sparse=self.sparse_inputs)

            rec = dot(intermediate_product, tf.transpose(m2g), sparse=False)

            supports.append(rec)
        output = tf.add_n(supports)
        # output = rec
        output = tf.nn.l2_normalize(output, dim=1)
        output = self.act(output)
        return output
        # return tf.nn.sigmoid(output)
        # return tf.nn.softmax(output)