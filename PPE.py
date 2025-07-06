import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.colors import ListedColormap
import time


class DEMGenerator:
    def __init__(self, shape, *args):
        #shape 包含了地图的单元格数量
        #args 元组，包含了山峰 （位置、高度、范围）
        self.map_shape = shape
        self.peak_desc = args
        self.X = np.zeros((100,100), dtype='f4')
        self.Y = np.zeros((100,100), dtype='f4')
        self.Z = np.zeros((100,100), dtype='f4')

    # 创建山峰的函数
    def mountain(self, x, y, x0, y0, h, r):
        return np.abs(h * np.exp(-((x-x0)**2 + (y-y0)**2) / r**2))

    def gemDEM(self):
        x = np.linspace(0, self.map_shape[0], self.map_shape[0], dtype='i4')
        y = np.linspace(0, self.map_shape[1], self.map_shape[1], dtype='i4')

        self.X, self.Y = np.meshgrid(x, y)


        for desc in self.peak_desc:
            self.Z  += self.mountain(self.X, self.Y, desc[0], desc[1], desc[2], desc[3])
            # 添加一些随机噪声使地形更自然
            self.Z += np.random.normal(5, 0.2, self.Z.shape)
        return self.Z

    def plt_DEM(self):
        # 创建3D图形
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制表面
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap='terrain', edgecolor='none')

        # 添加颜色条
        fig.colorbar(surf, shrink=0.5, aspect=5, label='height')

        # 设置标签和标题
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('height')
        ax.set_title('Grid Map With Some Peaks')
        # 显示图形
        plt.tight_layout()

        plt.figure(2)

        plt.contourf(self.X, self.Y, self.Z, 20, cmap='terrain')
        plt.colorbar(label='height')
        plt.title('DEM Top View')
        plt.grid(True)


        plt.show()


class Astar():
    def __init__(self, map, s, e):
        self.safe_height = 10.0 # 安全高度
        self.map = map
        self.start = s
        self.goal  = e
        self.plt = False

        self.raws = self.map.shape[0]
        self.columes = self.map.shape[1]


    def neighbors(self, node):
        nb = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        return [self.Node((node.pos + p)) for p in nb]

    def cost(self, pos):
        g = 0.0
        h = 0.0
        for i in range(len(pos)):
            g += abs(pos[i] - self.start[i])
            h += abs(pos[i] - self.goal[i])
        return g + h

    def accessible(self, node):
        pos = node.pos

        is_in_map = pos[0] >= 0 and pos[0] < self.raws and pos[1] >= 0 and pos[1] < self.columes
        if is_in_map == False:
            return False
        return is_in_map and self.map[pos[0]][pos[1]] + self.safe_height < pos[2]

    class Node():
        def __init__(self, pos):
            self.pos = pos
            self.g = None
            self.h = None
            self.cost = None
            self.parent = None

        def __lt__(self, other):
            return self.cost < other.cost

        def __eq__(self, other):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1] and self.pos[2] == other.pos[2]

        def __hash__(self):
            return hash((self.pos[0], self.pos[1], self.pos[2]))

    def plot_path(self, path):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the DEM surface
        X, Y = np.meshgrid(np.arange(self.map.shape[0]), np.arange(self.map.shape[1]))
        ax.plot_surface(X, Y, self.map, cmap='terrain', alpha=0.5)

        # Plot the path
        if path:
            path_x = [node.pos[0] for node in path]
            path_y = [node.pos[1] for node in path]
            path_z = [node.pos[2] for node in path]
            ax.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Path')
            ax.scatter(path_x, path_y, path_z, c='r', s=50)

        # Mark start and goal
        ax.scatter(self.start[0], self.start[1], self.start[2], c='g', s=100, marker='*', label='Start')
        ax.scatter(self.goal[0], self.goal[1], self.goal[2], c='b', s=100, marker='*', label='Goal')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Height')
        ax.set_title('3D Path Planning with A* Algorithm')
        ax.legend()
        plt.show()

    def search(self, visualize=False):
        start_node = self.Node(self.start)
        end_node   = self.Node(self.goal)
        start_node.cost = self.cost(start_node.pos)

        open_set  = []
        close_set = set()
        heapq.heappush(open_set, start_node)

        while open_set:
            cur_node = heapq.heappop(open_set)

            #到终点后回溯返回路径
            if cur_node == end_node:
                path = []
                path.append(cur_node)
                while cur_node.parent is not None:
                    cur_node = cur_node.parent
                    path.append(cur_node)

                print("*****PATH ALREADY FOUND!!!*****")
                if visualize:
                    self.plot_path(path)
                return path[::-1]

            # 放入闭集
            close_set.add(cur_node)
            for neighbor in self.neighbors(cur_node):
                if self.accessible(neighbor) == False or neighbor in close_set:
                    continue

                neighbor.cost = self.cost(neighbor.pos)
                neighbor.parent = cur_node
                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
        print("*****NO PATH!!!*****")
        return []


if __name__ == "__main__":
    dem = DEMGenerator((100,100),
                       (15, 15, 30, 10),
                       (30, 50, 100, 25),
                       (80, 80, 80, 15),
                       (80, 40, 60, 20))
    map = dem.gemDEM()
    # dem.plt_DEM()

    astar = Astar(map, (0,0,50), (10,10,100))

    path = astar.search(True)
    # print(path)



