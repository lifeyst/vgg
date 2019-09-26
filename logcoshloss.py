
import tensorflow as tf



def get_logcosh_loss(features,batch_size):

    len_features = features.get_shape()[1]
    centers1 = tf.get_variable('centers1', [batch_size, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True)
    b = tf.slice(centers1, [1,0], [batch_size-1,len_features])
    logcosh_loss=tf.reduce_mean(tf.log(tf.cosh(features[0]-tf.reduce_mean(b,0))))
 
    return logcosh_loss
