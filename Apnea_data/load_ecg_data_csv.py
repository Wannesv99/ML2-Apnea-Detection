import pandas as pd
import numpy as np

data_df = pd.read_csv('./'+physionet_folder+'/a01.csv',header=None)
data_arr = data_df.to_numpy()
print("Shape of array: ",arr.shape)
print("Type of array: ",type(arr))


