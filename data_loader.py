import pandas as pd

reader = pd.read_csv('data/train_messages.csv')

small_reader = reader[0:150000]

small_reader.to_csv('data/small_train_messages.csv', index=False)