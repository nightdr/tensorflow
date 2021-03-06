import pandas as pd
import tensorflow as tf

# retrieve data

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

inputX = dataframe.loc[:, ["X", "Y"]].as_matrix()
inputY = dataframe.loc[:, ["Z"]].as_matrix()

print(inputX)

print(inputY)



# Hyper Parameters
learning_rate = 1
training_epochs = 1000
display_step = 10


# graph inputs
x = tf.placeholder(tf.float32, [None, 2], name = "x") # x and y inputs
y = tf.placeholder(tf.float32, [None, 2], name = "labels") # z output as one hot array

# want model to output either 0 or 1
"""

W1 = tf.Variable(tf.zeros([2, 2]), name = "W1")
b1 = tf.Variable(tf.zeros([2]), name = "B2")

W2 = tf.Variable(tf.zeros([2, 2]), name = "W2")
b2 = tf.Variable(tf.zeros([2]), name = "B2")

"""

W1 = tf.Variable(tf.random_normal([2, 2]), name = "W1")
b1 = tf.Variable(tf.random_normal([2]), name = "B2")

W2 = tf.Variable(tf.random_normal([2, 2]), name = "W2")
b2 = tf.Variable(tf.random_normal([2]), name = "B2")


# tf.summary.histogram("W", W)
# tf.summary.histogram("b", b)


# model
layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

logits = tf.matmul(layer1, W2) + b2

pred = tf.nn.softmax(logits)


# cost function
with tf.name_scope("cost"):
    # cost = tf.reduce_sum(tf.pow((pred - y), 2))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
    # cost = tf.losses.mean_squared_error(labels = y, predictions = logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar("cost", cost)

init = tf.global_variables_initializer()

# running graph
with tf.Session() as sess:
    # flatten input Y then convert it to a one hot matrix
    inputY = sess.run(tf.reshape(inputY, [4]))
    inputY = sess.run(tf.one_hot(inputY, 2))

    print(inputY)

    sess.run(init)

    writer = tf.summary.FileWriter("E:\\Tensorflow\\nn\\4")
    writer.add_graph(sess.graph)

    merged_summary = tf.summary.merge_all()

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={x: inputX, y: inputY})

        if epoch % display_step == 0:
            outCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
            outCost = sess.run(tf.reduce_mean(outCost))
            print("step: %04d" % epoch, "cost =  %.9f" % outCost)
            # s = sess.run(merged_summary, feed_dict={x: inputX, y: inputY})
            # writer.add_summary(s, epoch)

    print("Done")

    finalCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
    finalCost = sess.run(tf.reduce_mean(finalCost))

    print("Final Cost = ", finalCost, "weights = ", sess.run(W1), sess.run(W2), "biases = ", sess.run(b1), sess.run(b2))


    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    print(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}))
    accuracy = tf.reduce_mean(tf.cast(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}), tf.float32))
    print("Accuracy:", accuracy.eval({x: inputX, y: inputY}))

    print(sess.run(pred, feed_dict= {x: inputX}))
    print(sess.run(tf.argmax(pred, 1), feed_dict = {x: inputX}))

