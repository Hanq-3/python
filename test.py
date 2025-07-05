import numpy as np

import queue


# 创建一个 3x3 的二维数组
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(array[:, 2])


pq = queue.PriorityQueue()


pq.put((11, [1,1,1]))
pq.put((99, (9,9,9)))

pq.put((0, (0,0,0)))

pq.put((55, (5,5,5)))



while not pq.empty():
    print(pq.get())



