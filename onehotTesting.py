import pandas as pd
import tensorflow as tf

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

inputX = dataframe.loc[:, ["X", "Y"]].as_matrix()
inputY = dataframe.loc[:, ["Z"]].as_matrix()

print(inputX)

print(inputY)

reshaped = tf.Session().run(tf.reshape(inputY, [4]))

# print(tf.Session().run(tf.one_hot(inputY, depth)))

print(reshaped)

one_hot = tf.Session().run(tf.one_hot(reshaped, 2))

print(one_hot)