import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# x_values = [203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146]
# y_values = [495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344]
# sigma_y  = [21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22]

data = pd.read_csv("Paper_Exercises\data.csv")


# Drop the first four rows
exs1_data = data.iloc[4:]

final_data= exs1_data.drop(["sigma_x", "rho_xy"], axis=1)
print(final_data)