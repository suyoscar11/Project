import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Paper_Exercises\data.csv")


x= data["x"].values
y = data["y"].values
sigma_y=data["sigma_y"].values
sigma_x=data["sigma_x"].values
rho_xy= data["rho_xy"].values

Y = y.reshape(-1,1)


#.vstack uses to stack arrays on top of one another, np.ones_like creates 1's in the 1st columns, then x on the second ones
A = np.vstack((np.ones_like(x),x)).T

#Could have used np.zeros, but just a diagonal matrix would work
C = np.diag(sigma_y**2)

#Calculating [A.transpose C.inverse A].inverse (The first part)
#Also another thing about matrix multiplication, in numpy, we use this @ to multiply

left_part = np.linalg.inv(A.T @ np.linalg.inv(C) @ A)
right_part= A.T @ np.linalg.inv(C) @ Y

X= left_part @ right_part

b,m = X.flatten()

variance= left_part[1,1]

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Variance of slope (sigma_m^2): {variance}")

# Plot data points with uncertainties and best-fit line
plt.errorbar(x, y, yerr=sigma_y, fmt='o', label='Data with uncertainties', capsize=4)
plt.plot(x, m * x + b, label=f'Fit line: y={m:.2f}x + {b:.2f}', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Fit with Uncertainties')
plt.legend()
plt.grid(True)
plt.show()
