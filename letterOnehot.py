import tensorflow as tf
import pandas as pd

df = pd.read_csv("C:/Users/David/Desktop/Letter Grid/labels.csv")

derivedLabels = df.loc[:, "A"].as_matrix()

alphabet = 'abcdefghijklmnopqrstuvwxyz'

# converts a -> 0, b -> 1, etc.
char_to_int = dict((c, i) for i, c in enumerate(derivedLabels))

stringInts = [char_to_int[char] for char in alphabet]

one_hot = tf.Session().run(tf.one_hot(stringInts, 26))