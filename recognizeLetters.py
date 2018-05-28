import pandas as pd
import tensorflow as tf
from PIL import Image

# Hyper Parameters
learning_rate = 0.1
training_epochs = 70
display_step = 10


# fetch and format data
inputX = []

for i in range(1, 27):

    im = Image.open("C:\\Users\\David\\Desktop\\Letter Grid\\Letters\\Times\\%d.png" % i)

    imgData = list(im.getdata())

    avgList = []

    for index in range(0, len(imgData)):
        r, g, b = imgData[index]
        avg = int((r + g + b) / 3)
        avgList.append(avg)

    inputX.append(avgList)

print(inputX)

# format labels

df = pd.read_csv("C:/Users/David/Desktop/Letter Grid/labels.csv")

derivedLabels = df.loc[:, "A"].as_matrix()

alphabet = 'abcdefghijklmnopqrstuvwxyz'

# converts a -> 0, b -> 1, etc.
char_to_int = dict((c, i) for i, c in enumerate(derivedLabels))

stringInts = [char_to_int[char] for char in alphabet]

inputY = tf.Session().run(tf.one_hot(stringInts, 26))

print(inputY)

# training data

# fetch and format data
testX = []

for i in range(1, 27):

    im = Image.open("C:\\Users\\David\\Desktop\\Letter Grid\\Letters\\Arial\\%d.png" % i)

    imgData = list(im.getdata())

    avgList = []

    for index in range(0, len(imgData)):
        r, g, b = imgData[index]
        avg = int((r + g + b) / 3)
        avgList.append(avg)

    testX.append(avgList)

print(testX)

# format labels

df = pd.read_csv("C:/Users/David/Desktop/Letter Grid/labels.csv")

derivedLabels = df.loc[:, "A"].as_matrix()

alphabet = 'abcdefghijklmnopqrstuvwxyz'

# converts a -> 0, b -> 1, etc.
char_to_int = dict((c, i) for i, c in enumerate(derivedLabels))

stringInts = [char_to_int[char] for char in alphabet]

testY = tf.Session().run(tf.one_hot(stringInts, 26))

print(testY)


# graph inputs
x = tf.placeholder(tf.float32, [None, 2500], name = "x") # pixel inputs 50 x 50 = 2500
y = tf.placeholder(tf.float32, [None, 26], name = "labels") # labels as one hot arrays, 26 letters in the alphabet

W1 = tf.Variable(tf.random_normal([2500, 2500]), name = "W1")
b1 = tf.Variable(tf.random_normal([2500]), name = "B2")

W2 = tf.Variable(tf.random_normal([2500, 26]), name = "W2")
b2 = tf.Variable(tf.random_normal([26]), name = "B2")


# model
layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

logits = tf.matmul(layer1, W2) + b2

pred = tf.nn.softmax(logits)


# cost function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
            outCost = sess.run(tf.reduce_mean(outCost))
            print("step: %04d" % epoch, "cost =  %.9f" % outCost)
            # s = sess.run(merged_summary, feed_dict={x: inputX, y: inputY})
            # writer.add_summary(s, epoch)

    print("Done")

    finalCost = sess.run(cost, feed_dict={x: inputX, y: inputY})
    finalCost = sess.run(tf.reduce_mean(finalCost))

    print("Final Cost = ", finalCost)
    print("weights = ", sess.run(W1), sess.run(W2))
    print("biases = ", sess.run(b1), sess.run(b2))


    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    print(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}))
    accuracy = tf.reduce_mean(tf.cast(sess.run(correct_prediction, feed_dict= {x: inputX, y: inputY}), tf.float32))
    print("Accuracy:", accuracy.eval({x: inputX, y: inputY}))

    #print(sess.run(pred, feed_dict= {x: inputX}))
    print(sess.run(tf.argmax(pred, 1), feed_dict = {x: inputX}))

    print(sess.run(correct_prediction, feed_dict= {x: testX, y: testY}))
    accuracy = tf.reduce_mean(tf.cast(sess.run(correct_prediction, feed_dict= {x: testX, y: testY}), tf.float32))
    print("Accuracy:", accuracy.eval({x: testX, y: testY}))

    print(sess.run(tf.argmax(pred, 1), feed_dict = {x: testX}))