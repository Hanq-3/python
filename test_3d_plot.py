import pandas as pd
import numpy as np
from numpy.ma.extras import polyfit
import matplotlib.pyplot as plt

x = np.linspace(10, 30, 100, dtype=np.float32)
y = x**3 + 2*x**2 + 5


coef = polyfit(x, y, 5)
print(coef)



expression = np.poly1d(coef)

y_fit = expression(x)

print(y_fit)

fig = plt.figure(0)

ax = fig.add_subplot(111)
ax.set_title("ax")
ax.plot(x, y, 'r*', x, y_fit, 'b.')


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.set_title("ay")
ax1.plot(x, y_fit,'b.')

plt.figure(3)
plt.xlabel("f3 x")
plt.plot(x, y, 'g+')

plt.show()



