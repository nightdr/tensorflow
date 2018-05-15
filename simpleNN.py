import pandas as pd
import tensorflow as tf

# Retrieve data

import tensorflow as tf

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

inputX = dataframe.loc[:, ["X", "Y"]].as_matrix()
inputY = dataframe.loc[:, ["Z"]].as_matrix()

print(inputX)

print(inputY)



# Hyper Parameters
learning_rate = 0.1
training_epochs = 100
n_samples = inputY.size
display_step = 1


# graph inputs
x = tf.placeholder(tf.float32, [None, 2], name = "x") # x and y inputs
y = tf.placeholder(tf.float32, [None, 1], name = "labels") # z output

W = tf.Variable(tf.zeros([2, 1]), name = "W")
b = tf.Variable(tf.zeros([1]), name = "B")

tf.summary.histogram("W", W)
tf.summary.histogram("b", b)


# model
logits = tf.matmul(x, W) + b

pred = tf.nn.sigmoid(logits)


# cost function
with tf.name_scope("cost"):
    cost = tf.reduce_sum(tf.pow((pred - y), 2))
    # cost = tf.pow(tf.subtract(pred, y), 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar("cost", cost)

init = tf.global_variables_initializer()

# running graph
with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("E:\\Tensorflow\\nn\\4")
    writer.add_graph(sess.graph)

    merged_summary = tf.summary.merge_all()

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={x: inputX, y: inputY})

        if epoch % display_step == 0:
            outCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
            print("training, step: %04d" % epoch, "cost =  %.9f" % outCost)
            # s = sess.run(merged_summary, feed_dict={x: inputX, y: inputY})
            # writer.add_summary(s, epoch)

    print("Done")

    finalCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
    print("Final Cost = ", finalCost, "w = ", sess.run(W), "b = ", sess.run(b))


    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    print(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}), tf.float32)
    accuracy = tf.reduce_mean(tf.cast(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}), tf.float32))
    print("Accuracy:", accuracy.eval({x: inputX, y: inputY}))

    print(sess.run(pred, feed_dict= {x: inputX}))

