from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.node_feat= []
        self.m2g_feat_layers = []
        self.d2g_feat_layers = []
        self.d2m_layer = None
        self.inputs = None
        self.output_m2g = None
        self.output_d2g = None
        self.outputs = None
        self.loss_pres = None
        self.loss_labs = None
        self.loss_pres_visual = None
        self.loss_labs_visual =None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.total = None
        self.test = None
        self.negative_sample = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        result_m2g = []
        result_d2g = []
        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            self.node_feat.append(hidden)

        # for i, layer in self.m2g_feat_layers:
        #     result.append(layer(self.node_feat[i]))

        for i in range(len(self.m2g_feat_layers)):
            if i==0:
                result_m2g.append( self.m2g_feat_layers[0](self.inputs) )
            else:
                result_m2g.append( self.m2g_feat_layers[i](self.node_feat[i-1]) )
        self.output_m2g = tf.reduce_mean(result_m2g, axis=0)

        for i in range(len(self.d2g_feat_layers)):
            if i==0:
                result_d2g.append( self.d2g_feat_layers[0](self.inputs) )
            else:
                result_d2g.append( self.d2g_feat_layers[i](self.node_feat[i-1]) )
        self.output_d2g = tf.reduce_mean(result_d2g, axis=0)

        self.total = tf.concat([self.output_d2g,self.output_m2g], 0)
        self.outputs = self.d2m_layer(self.total)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self.negative_sample = self._negative_samples()
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features_g']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # self.output_dim = placeholders['labels_d2r_adj'].get_shape().as_list()[1]

        self.adj_m2g = placeholders['adj_m2g']
        self.adj_d2g = placeholders['adj_d2g']
        self.adj_d2m = placeholders['adj_d2m']
        self.mirna_length = placeholders['mirna_length']
        self.diease_length = placeholders['diease_length']
        self.labels_row = placeholders['labels_row']
        self.positive_train_label_row = placeholders['positive_train_label_row']
        self.neg_train_label_row = placeholders['neg_train_label_row']
        self.output_dim = 32
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.neg_output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        # self.train_edge = placeholders['train_edge']
        # self.features_diease = placeholders['features_diease']
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for index in range(3):
            for var in self.m2g_feat_layers[index].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for var in self.d2g_feat_layers[index].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in self.d2m_layer.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss_pres_visual = tf.gather(self.outputs, self.labels_row, axis=0)
        self.loss_labs_visual = tf.gather(self.adj_d2m, self.labels_row, axis=0)

        positive_pres = tf.gather_nd(self.outputs, self.positive_train_label_row)
        positive_labs = tf.gather_nd(self.adj_d2m, self.positive_train_label_row)

        neg_pres = tf.gather_nd(self.outputs, self.negative_sample)
        neg_labs = tf.gather_nd(self.adj_d2m, self.negative_sample)

        elems_pres = positive_pres,neg_pres
        loss_pres_con = tf.map_fn(self.get_sample_pres,elems_pres,dtype="float")
        elems_labs = positive_labs,neg_labs
        loss_labs_con = tf.map_fn(self.get_sample_labs,elems_labs,dtype="float")

        # loss_pres_con = tf.concat([positive_pres, neg_pres], 0)
        # loss_labs_con = tf.concat([positive_labs, neg_labs], 0)
        # loss_pres_labs = tf.random_shuffle([loss_pres_con,loss_labs_con])
        # self.loss_pres = loss_pres_labs[:,0]
        # self.loss_labs = loss_pres_labs[:,1]
        # self.loss_labs = tf.random_shuffle(loss_labs_con)
        self.loss_pres = tf.reshape(loss_pres_con,(-1,1))
        self.loss_labs = tf.reshape(loss_labs_con,(-1,1))

        # c0 = tf.constant([2, 4, 6])
        # print(c0.shape[0])
        # stop = tf.cast(self.loss_pres.shape[0], tf.float32)
        stop = tf.cast(tf.shape(self.loss_pres)[0], tf.float32)
        a = tf.linspace(0.0, stop - 1, tf.shape(self.loss_pres)[0])
        a = tf.cast(a, tf.int32)
        aa = tf.random_shuffle(a)
        self.loss_pres = tf.gather(self.loss_pres, aa)
        self.loss_labs = tf.gather(self.loss_labs, aa)

        # Cross entropy error
        self.test = masked_softmax_cross_entropy(self.loss_pres, self.loss_labs)
        self.loss += self.test
        tf.add_to_collection('losses',self.loss)
        self.loss = tf.add_n(tf.get_collection('losses'))

    def get_sample_labs(self,input):
        a = input[0]
        b = input[1]
        return tf.stack([a, b], 0)

    def get_sample_pres(self,input):
        a = input[0]
        b = input[1]
        return tf.stack([a, b], 0)

    def get_positive_dis(self, input):
        edge = input
        list_edge = tf.unstack(edge, axis=0)
        dis = tf.linalg.norm(self.output_d2g[list_edge[0]] - self.output_m2g[list_edge[1]])
        return dis

    def get_positive_dis_vector(self, input):
        edge = input
        dis_vector = tf.zeros((32,))
        list_edge = tf.unstack(edge, axis=0)
        list_edge = tf.cast(list_edge, dtype=tf.int32)
        dis_vector = self.output_d2g[list_edge[0]] - self.output_m2g[list_edge[1]]
        return dis_vector

    def _negative_samples(self):
        average_positive_train_dis = 0.0
        elems = self.positive_train_label_row
        dis = tf.map_fn(self.get_positive_dis, elems, dtype="float")
        dis_vector = tf.map_fn(self.get_positive_dis_vector,elems,dtype="float")
        # average_positive_train_dis = tf.reduce_sum(dis) / self.positive_train_label_row.get_shape().as_list()[0]
        # average_positive_train_vector = tf.reduce_sum(dis_vector, axis=0) / self.positive_train_label_row.get_shape().as_list()[0]
        train_label_len = tf.shape(self.positive_train_label_row)[0]
        average_positive_train_dis = tf.reduce_sum(dis) / tf.cast(train_label_len,tf.float32)
        average_positive_train_vector = tf.reduce_sum(dis_vector, axis=0) / tf.cast(train_label_len,tf.float32)

        neg_dis = tf.Variable([])
        neg_output = tf.Variable([])
        n = tf.constant(0)
        x = tf.shape(self.neg_train_label_row)[0]
        step = tf.constant(0)
        # neg_output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        def fn_edge_1(a, b, neg_output):
            # global  neg_output
            # neg_output = neg_output.write(step, (a, b))
            neg_output = tf.concat([neg_output, [a,b]], 0)
            return neg_output #tf.constant(1)
        def fn_edge_2(a, b, neg_output):
            return neg_output #tf.constant(0)

        def fn_dis_1(a,neg_dis):
            # neg_dis.append(a)
            neg_dis = tf.concat([neg_dis, [a]], 0)
            return neg_dis
        def fn_dis_2(a,neg_dis):
            return neg_dis

        def for_body(n,neg_dis):
            n_edge = tf.unstack(self.neg_train_label_row[n])
            n_dis = tf.linalg.norm(self.output_d2g[n_edge[0]] - self.output_m2g[n_edge[1]])
            condition = tf.greater(n_dis, average_positive_train_dis)
            # tf.cond( condition, lambda: fn_edge_1(n_edge[0], n_edge[1], neg_dis_edge), lambda: fn_edge_2(n_edge[0], n_edge[1], neg_dis_edge))
            neg_dis = tf.cond( condition, lambda: fn_dis_1(n_dis,neg_dis), lambda: fn_dis_2(n_dis,neg_dis))
            n = n+1
            return n,neg_dis
        def for_condition(n,neg_dis):
            return n<x
        n,neg_dis = tf.while_loop(for_condition, for_body, [n,neg_dis], shape_invariants=[n.get_shape(),tf.TensorShape([None])])

        def for_body_edge(step, neg_output):
            n_edge = tf.unstack(self.neg_train_label_row[step])
            n_dis = tf.linalg.norm(self.output_d2g[n_edge[0]] - self.output_m2g[n_edge[1]])
            condition = tf.greater(n_dis, average_positive_train_dis)
            neg_output = tf.cond(condition, lambda: fn_edge_1(n_edge[0], n_edge[1], neg_output), lambda: fn_edge_2(n_edge[0], n_edge[1], neg_output))
            # output = output.write(step, tf.gather(array, step))
            step = step+1
            return step, neg_output
        def for_condition_edge(step, neg_output):
            return step<x
        _, neg_dis_edge = tf.while_loop(for_condition_edge, for_body_edge, loop_vars=[step, neg_output], shape_invariants=[step.get_shape(),tf.TensorShape([None])])

        # for neg in range(self.neg_train_label_row.get_shape().as_list()[0]):
        #     n_edge = tf.unstack(self.neg_train_label_row[neg])
        #     n_dis = tf.linalg.norm(self.output_d2g[n_edge[0]] - self.output_m2g[n_edge[1]])
        #     condition = tf.greater(n_dis, average_positive_train_dis)
        #     # tf.cond( (n_dis > average_positive_train_dis), lambda:neg_dis_edge.append((n_edge[0], n_edge[1])), lambda:no_use_edge.append((n_edge[0], n_edge[1])) )
        #     # tf.cond( (n_dis > average_positive_train_dis), lambda:neg_dis.append(n_dis), lambda:no_use_dis.append(n_dis) )
        #     tf.cond( condition, lambda: fn_edge_1(n_edge[0], n_edge[1]), lambda: fn_edge_2(n_edge[0], n_edge[1]))
        #     tf.cond( condition, lambda: fn_dis_1(n_dis), lambda: fn_dis_2(n_dis))
        neg_dis = tf.stack(neg_dis)
        neg_dis_edge = tf.cast(neg_dis_edge,tf.int32)
        neg_dis_edge = tf.stack(neg_dis_edge)
        neg_dis_edge = tf.reshape(neg_dis_edge, (-1, 2))
        # k_max_dis = tf.nn.top_k(neg_dis,self.positive_train_label_row.get_shape().as_list()[0])
        k_max_dis = tf.nn.top_k(neg_dis, tf.shape(self.positive_train_label_row)[0])
        k_max_dis_value = k_max_dis[0]
        k_max_dis_idx = k_max_dis[1]
        neg_k_max_edge = tf.gather(neg_dis_edge, k_max_dis_idx, axis=0)
        return neg_k_max_edge


    def _accuracy(self):
        self.accuracy = masked_accuracy(tf.nn.sigmoid(self.loss_pres), self.loss_labs)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        self.m2g_feat_layers.append(EdgeDecoder(idx=0, input_dim=self.input_dim,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.relu,
                                                 dropout=True,
                                                 sparse_inputs=True,
                                                 logging=self.logging))
        self.m2g_feat_layers.append(EdgeDecoder(idx=0, input_dim=FLAGS.hidden1,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.relu,
                                                 dropout=True,
                                                 logging=self.logging))
        self.m2g_feat_layers.append(EdgeDecoder(idx=0, input_dim=FLAGS.hidden2,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: x,
                                                 dropout=True,
                                                 logging=self.logging))

        self.d2g_feat_layers.append(EdgeDecoder(idx=1, input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
        self.d2g_feat_layers.append(EdgeDecoder(idx=1, input_dim=FLAGS.hidden1,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))
        self.d2g_feat_layers.append(EdgeDecoder(idx=1, input_dim=FLAGS.hidden2,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

        self.d2m_layer = (d2m_EdgeDecoder(input_dim=self.output_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))


    def predict(self):
        return tf.nn.sigmoid(self.outputs)


