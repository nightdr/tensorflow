import tensorflow as tf

def train_input(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


# Feature columns describe how to use the input.
feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

def my_model(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, params["feature_columns"])

    d1 = tf.layers.dense(input_layer, 2, activation = tf.nn.relu)

    logits = tf.layers.dense(d1, 1)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids' : predicted_classes[:, tf.newaxis],
            'probabilities' : tf.nn.softmax(logits),
            'logits' : logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    accuracy = tf.metrics.accuracy(labels = labels, predictions = predicted_classes, name = "acc_op")

    metrics = {"accuracy" : accuracy}
    tf.summary.scalar("accuracy", accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
        train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

classifier = tf.estimator.Estimator(
    model_fn = my_model,
    params = {
        "feature_columns" : feature_columns
    }
)