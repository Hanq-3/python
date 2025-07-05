import csv
import re
import numpy as np
import matplotlib.pyplot as plt

with open('1.csv', encoding='utf-8') as file:
    csv_file = csv.reader(file)
    print(type(csv_file))

    for row in csv_file:
        print(type(row))
        print(type(row[0]))
        print(row)




plt.figure(1)
plt.plot(np.linspace(1,100,10), np.linspace(10,20,10))
# plt.show()


plt.figure(2)
plt.plot(np.linspace(1,100,10), np.linspace(10,30,10))
plt.show()