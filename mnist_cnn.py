from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as py
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    tf.app.run()

def cnn_model(features, labels, mode):

    #Layers

        #Inputs
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[5, 5],
        padding = "same",
        activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size= [2, 2],
        strides = 2
    )

    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2, 2],
        strides = 2
    )

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024,
        activation = tf.nn.relu
    )

    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(
        inputs = dropout,
        units = 10
    )

    predictions = {
        # Generate predictions
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(lables = labels, logits = logits)

    # Configure training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)







