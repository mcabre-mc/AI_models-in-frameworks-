import tensorflow_decision_forests as tfdf
import pandas as pd
import time

start = time.time()
data = pd.read_csv('/Users/anishapareek/Downloads/abalone.data')
end = time.time()
data_loading_time = end-start

num_of_columns = data.shape[1]
dataset = data.iloc[:, 1:num_of_columns-1]

start = time.time()
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset)
model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)
end = time.time()
training_time = end - start


print(model.summary())
