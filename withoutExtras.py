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
x = tf.placeholder(tf.float32, [None, 2]) # x and y inputs
y = tf.placeholder(tf.float32, [None, 1]) # z output

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))


# model
logits = tf.matmul(x, W) + b

pred = tf.nn.sigmoid(logits)



cost = tf.reduce_sum(tf.pow((pred - y), 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# running graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={x: inputX, y: inputY})

        if epoch % display_step == 0:
            outCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
            print("training, step: %04d" % epoch, "cost =  %.9f" % outCost)

    print("Done")

    finalCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
    print("Final Cost = ", finalCost, "w = ", sess.run(W), "b = ", sess.run(b))

    print(sess.run(pred, feed_dict= {x: inputX}))

