import numpy as np
import pandas as pd

df = pd.read_csv('data/train.csv', sep=chr(1), error_bad_lines=False)

number_of_files = 3
for id, df_i in enumerate(np.array_split(df, number_of_files)):
    df_i.to_csv('data/version/train_{id}.csv'.format(id=id))
