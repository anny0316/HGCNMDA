from __future__ import division
import tensorflow as tf




    # for dis in range(len(positive_train_label_row)):
    #     edge = list(positive_train_label_row[dis])
    #     op3 = op3 + (d2g[edge[0]] - m2g[edge[1]])
    #     op1 = op1 + np.linalg.norm((d2g[edge[0]] - m2g[edge[1]]))
    #     op2 = op2 + np.sqrt(np.sum(np.square(d2g[edge[0]] - m2g[edge[1]])))
    # average_positive_train_dis = op1 / len(positive_train_label_row)
    # average_positive_train_vector = op3 / len(positive_train_label_row)


def get_consine_simi(input):
    a = input[0]
    b = input[1]
    c = 0.0
    dot_product = tf.multiply(a, b)
    sum = tf.reduce_sum(dot_product)
    c=tf.where(tf.equal(sum, 0.0), 0.000001, sum / (tf.linalg.norm(a) * tf.linalg.norm(b)))
    return c

def masked_softmax_cross_entropy(preds, labels):  #preds, labels
    """Softmax cross-entropy loss with masking."""

    # elems = preds,labels
    # r_preds = tf.map_fn(get_consine_simi,elems,dtype="float")
    # r_labels = tf.where( tf.less(tf.reduce_sum(labels,1), 1.0), tf.zeros_like(r_preds), tf.ones_like(r_preds))
    #
    # # loss = tf.nn.softmax_cross_entropy_with_logits(logits=r_preds, labels=r_labels)
    # # optimization using the hinge loss
    # # diff = tf.nn.relu(tf.subtract(r_labels, tf.expand_dims(r_preds, 0)), name='diff')
    #
    # diff = tf.nn.relu(tf.subtract(r_labels, r_preds), name='diff')
    #
    # loss = tf.reduce_sum(diff)
    # return loss, r_preds, r_labels

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds,labels=labels)
    return tf.reduce_mean(loss)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # loss *= mask
    # return tf.reduce_mean(loss)

def get_knn(input):
    a = input[0]
    b = input[1]
    b = tf.cast(b, tf.int32)

    index = tf.where(tf.equal(b,1))
    index = tf.squeeze(index)
    index = tf.reshape(index, [-1])
    sum = tf.reduce_sum(b)
    sum = tf.cast(sum,tf.int32)
    location_idx = tf.nn.top_k(a,sum)[1]    #,False
    location_idx = tf.cast(location_idx,tf.int64)

    location_idx_sorted = tf.nn.top_k(location_idx,sum)[0]
    index_sorted = tf.nn.top_k(index,sum)[0]
    isequal = tf.reduce_all(tf.equal(location_idx_sorted,index_sorted))
    return tf.to_int32(isequal)

def masked_accuracy(preds, labels):
    """Accuracy with masking."""
    # preds_label = tf.Variable(tf.zeros_like(labels),validate_shape=False,name="preds_label")
    # elems = preds,labels
    # r_preds = tf.map_fn(get_knn, elems, dtype=tf.int32)
    # r_labels = tf.ones_like(r_preds)
    # return tf.reduce_sum(r_preds)/tf.reduce_sum(r_labels)
    r_preds = tf.where(tf.greater_equal(preds, 0.5), tf.ones_like(labels), tf.zeros_like(labels))
    correct_prediction = tf.equal(r_preds, labels)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

    # correct_prediction = tf.equal(tf.argmax(r_preds, 1), tf.argmax(r_labels, 1))
    # accuracy_all = tf.cast(correct_prediction, tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # accuracy_all *= mask
    # return tf.reduce_mean(accuracy_all)
