import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("Paper_Exercises\data.csv")


# Drop the first four rows
exs1_data = data.iloc[4:]

final_data= exs1_data.drop(["sigma_x", "rho_xy"], axis=1)
print(final_data)


#This .values will return a numpy array
x= final_data["x"].values
y= final_data["y"].values
sigma_y = final_data["sigma_y"].values

Y= y.reshape(-1,1)

A = np.vstack((np.ones_like(x),x, x**2)).T

C = np.diag(sigma_y**2)

left_part = np.linalg.inv(A.T @ np.linalg.inv(C) @ A)
right_part= A.T @ np.linalg.inv(C) @ Y

X= left_part @ right_part

b,m,q = X.flatten()

"""
[0,0] would be the variance of c (constant term)
[1,1] would be the variance of b (linear term)
[2,2] is the variance of a (quadratic term)
"""
variance= left_part[2,2]

print(f"Quadratic coefficient (q): {q}")
print(f"Linear coefficient (m): {m}")
print(f"Constant term (b): {b}")
print(f"Variance of quadratic coefficient (sigma_a^2): {variance}")
plt.errorbar(x, y, yerr=sigma_y, fmt='o', label='Data with uncertainties', capsize=4)
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = q * x_smooth**2 + m * x_smooth + b
plt.plot(x_smooth, y_smooth, label=f'Fit curve: y={q:.2f}x² + {m:.2f}x + {b:.2f}', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Fit with Uncertainties')
plt.legend()
plt.grid(True)
plt.show()

