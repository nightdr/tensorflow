import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# hyper parameters
learning_rate = 0.5
num_epochs = 1000
display_step = 20


# get data set
# one hot:
# 1 -> [0,1,0,0,0,0,0,0,0,0]
# 2 -> [0,0,1,0,0,0,0,0,0,0], etc.

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 28 * 28 = 784
# getting flattened image pixel values as input
x = tf.placeholder(tf.float32, shape=[None, 784])

#returning 1-9 number as output
y = tf.placeholder(tf.float32, shape=[None, 10])

# weight for each connection 784 -> 10
W = tf.Variable(tf.zeros([784,10]))
# bias for each node 10 (last output layer)
b = tf.Variable(tf.zeros([10]))

# model (multiply x by weights and add bias)
logits = tf.matmul(x, W) + b

# apply probability function
pred = tf.nn.softmax(logits)

max_pred = tf.reduce_max(pred)

# cost/loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)

for epoch in range(num_epochs):
    # grab 100 random training examples
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if epoch % display_step == 0:
        print("epoch: %d" % epoch, "current cost: %.4f" % sess.run(loss, feed_dict={x: batch_x, y: batch_y}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy: %.4f" % sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))