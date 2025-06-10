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


#This .values will return a numpy array
x= final_data["x"].values
y= final_data["y"].values
sigma_y = final_data["sigma_y"].values


#An interesting thing I found out while writing this code, so numpy has smthg like (N,1), where N is row and 1 is column, SInce according to the problem
#statement we need a column vector of Y, so the positional argument at index 2 will be column, and assigning -1 will automatically set up rows in numpy
Y = y.reshape(-1,1)


#.vstack uses to stack arrays on top of one another, np.ones_like creates 1's in the 1st columns, then x on the second ones
A = np.vstack((np.ones_like(x), x, x**2)).T  # Added x² term for quadratic fit

#Could have used np.zeros, but just a diagonal matrix would work
C = np.diag(sigma_y**2)

#Calculating [A.transpose C.inverse A].inverse (The first part)
#Also another thing about matrix multiplication, in numpy, we use this @ to multiply

left_part = np.linalg.inv(A.T @ np.linalg.inv(C) @ A)
right_part= A.T @ np.linalg.inv(C) @ Y

X= left_part @ right_part

c, b, a = X.flatten()  # Now we have three parameters: a, b, c for ax² + bx + c

variance= left_part[2,2]  # Variance of the quadratic term

print(f"Quadratic coefficient (a): {a}")
print(f"Linear coefficient (b): {b}")
print(f"Constant term (c): {c}")
print(f"Variance of quadratic coefficient (sigma_a^2): {variance}")

# Plot data points with uncertainties and best-fit curve
plt.errorbar(x, y, yerr=sigma_y, fmt='o', label='Data with uncertainties', capsize=4)

# Create a smooth curve for the quadratic fit
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = a * x_smooth**2 + b * x_smooth + c
plt.plot(x_smooth, y_smooth, label=f'Fit curve: y={a:.2f}x² + {b:.2f}x + {c:.2f}', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Fit with Uncertainties')
plt.legend()
plt.grid(True)
plt.show()
