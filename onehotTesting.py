import pandas as pd
import tensorflow as tf

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

inputX = dataframe.loc[:, ["X", "Y"]].as_matrix()
inputY = dataframe.loc[:, ["Z"]].as_matrix()

print(inputX)

print(inputY)


one_hot = [0,1,1,1]
depth = 2
reshaped = tf.Session().run(tf.reshape(inputY, [4, -1]))
# print(tf.Session().run(tf.one_hot(inputY, depth)))
print(tf.Session().run(tf.reshape(inputY, [4])))