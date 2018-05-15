import pandas as pd
import tensorflow as tf

dataframe = pd.read_csv("C:/Users/David/Desktop/orGateData.csv")

print(dataframe)

x = dataframe.loc[:, ["X", "Y"]].as_matrix()
y = dataframe.loc[:, ["Z"]].as_matrix()

print(x, y)