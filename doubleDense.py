import pandas as pd
import tensorflow as tf

# Retrieve data

import tensorflow as tf

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

inputX = dataframe.loc[:, ["X", "Y"]].as_matrix()
inputY = dataframe.loc[:, ["Z"]].as_matrix()

print(inputX)

print(inputY)



# hyper parameters
learning_rate = 1
training_epochs = 1000
n_samples = inputY.size
display_step = 1


# graph inputs
x = tf.placeholder(tf.float32, [None, 2]) # x and y inputs
y = tf.placeholder(tf.float32, [None, 1]) # z output

dW1 = tf.Variable(tf.zeros([2, 3]))
db1 = tf.Variable(tf.zeros([3]))

dW2 = tf.Variable(tf.zeros([3, 3]))
db2 = tf.Variable(tf.zeros([3]))

dW3 = tf.Variable(tf.zeros([3, 1]))
db3 = tf.Variable(tf.zeros([1]))

""" 
dW1 = tf.Variable(tf.zeros([2, 2]))
db1 = tf.Variable(tf.zeros([2]))

dW2 = tf.Variable(tf.zeros([2, 1]))
db2 = tf.Variable(tf.zeros([1]))

"""


# model

layer1 = tf.nn.sigmoid(tf.matmul(x, dW1) + db1)

layer2 = tf.nn.sigmoid(tf.matmul(layer1, dW2) + db2)

logits = tf.matmul(layer2, dW3) + db3

pred = tf.nn.sigmoid(logits)

# cost modeled after chi-squared calculation
cost = tf.reduce_sum(tf.pow((y - pred), 2)/pred)

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
    print("Final Cost = %.4f" % finalCost)
    print("\nWeights ", sess.run(dW1), " ", sess.run(dW2), "\n")
    print("biases ", sess.run(db1), " ", sess.run(db2), "\n")

    print("prediction: ")
    print(sess.run(pred, feed_dict= {x: inputX}))

