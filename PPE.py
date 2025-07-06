import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.colors import ListedColormap
import time
import bisect


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.entry_finder = {}  # 用于快速查找和替换节点

    def push(self, node):
        """添加或更新节点"""
        if node.pos in self.entry_finder:
            existing_node = self.entry_finder[node.pos]
            if node.g < existing_node.g:  # 只有新路径更优时才更新
                # 标记旧节点为已移除
                existing_node.removed = True
                # 添加新节点
                heapq.heappush(self.elements, node)
                self.entry_finder[node.pos] = node
        else:
            heapq.heappush(self.elements, node)
            self.entry_finder[node.pos] = node

    def pop(self):
        """弹出最小代价节点，跳过已标记移除的节点"""
        while self.elements:
            node = heapq.heappop(self.elements)
            if not hasattr(node, 'removed') or not node.removed:
                del self.entry_finder[node.pos]
                return node
        raise IndexError("pop from empty priority queue")

    def __contains__(self, pos):
        """检查位置是否在队列中"""
        return pos in self.entry_finder

    def __len__(self):
        """返回队列中有效节点的数量"""
        return len(self.entry_finder)

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
        self.nb = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        self.raws = self.map.shape[0]
        self.columes = self.map.shape[1]


    def neighbors(self, node):
        return [self.Node((node.pos + p)) for p in self.nb]

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
        return is_in_map and self.map[pos[0]][pos[1]] + self.safe_height < pos[2]

    class Node():
        def __init__(self, pos):
            self.pos = tuple(pos)
            self.g = float('inf')
            self.h = 0
            self.cost = float('inf')
            self.parent = None

        def __lt__(self, other):
            return self.cost < other.cost

        def __eq__(self, other):
            return self.pos == other.pos

        def __hash__(self):
            return hash(self.pos)

        # def __eq__(self, other):
        #     return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1] and self.pos[2] == other.pos[2]

        # def __hash__(self):
        #     return hash((self.pos[0], self.pos[1], self.pos[2]))

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

    def plot_path_with_profile(self, path):
        """绘制3D路径和高度剖面对比图"""
        fig = plt.figure(figsize=(16, 6))

        # 3D路径图
        ax1 = fig.add_subplot(121, projection='3d')
        X, Y = np.meshgrid(np.arange(self.map.shape[0]), np.arange(self.map.shape[1]))
        ax1.plot_surface(X, Y, self.map, cmap='terrain', alpha=0.5)

        if path:
            path_x = [node.pos[0] for node in path]
            path_y = [node.pos[1] for node in path]
            path_z = [node.pos[2] for node in path]

            # 绘制路径
            ax1.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Path')
            ax1.scatter(path_x, path_y, path_z, c='r', s=50)

            # 标记危险区域（路径高度接近地形高度）
            for node in path:
                x, y, z = node.pos
                terrain_z = self.map[int(x), int(y)]
                if z < terrain_z + self.safe_height:
                    ax1.scatter([x], [y], [z], c='black', s=100, marker='x', linewidths=2)

        # 标记起点和终点
        ax1.scatter(*self.start, c='g', s=100, marker='*', label='Start')
        ax1.scatter(*self.goal, c='b', s=100, marker='*', label='Goal')
        ax1.set_title('3D Path Visualization')
        ax1.legend()

        # 高度剖面图
        ax2 = fig.add_subplot(122)
        if path:
            # 计算路径距离
            path_array = np.array([node.pos for node in path])
            distances = np.cumsum(np.sqrt(np.sum(np.diff(path_array[:,:2], axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)

            # 获取地形高度（路径经过的点）
            terrain_heights = [self.map[int(p[0]), int(p[1])] for p in path_array]

            # 绘制高度对比
            ax2.plot(distances, path_array[:,2], 'r-', label='Path Height', linewidth=2)
            ax2.plot(distances, terrain_heights, 'b-', label='Terrain Height', linewidth=2)
            ax2.fill_between(distances, terrain_heights, path_array[:,2],
                            where=(path_array[:,2] > terrain_heights),
                            color='green', alpha=0.3, label='Safe Area')
            ax2.fill_between(distances, terrain_heights, path_array[:,2],
                            where=(path_array[:,2] <= terrain_heights),
                            color='red', alpha=0.3, label='Danger Area')

            # 标记安全高度阈值线
            ax2.plot(distances, np.array(terrain_heights)+self.safe_height,
                    'g--', label='Safe Height Threshold')

            ax2.set_xlabel('Path Distance (units)')
            ax2.set_ylabel('Height (units)')
            ax2.set_title('Path Height Profile')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def search(self, visualize=False):
        start_node = self.Node(self.start)
        end_node   = self.Node(self.goal)
        start_node.cost = self.cost(start_node.pos)

        open_set  = PriorityQueue()
        close_set = set()
        open_set.push(start_node)

        while open_set:
            cur_node = open_set.pop()

            #到终点后回溯返回路径
            if cur_node == end_node:
                path = []
                path.append(cur_node)
                while cur_node.parent is not None:
                    cur_node = cur_node.parent
                    path.append(cur_node)

                print("*****PATH ALREADY FOUND!!!*****")
                if visualize:
                    self.plot_path_with_profile(path)
                return path[::-1]

            # 放入闭集
            close_set.add(cur_node.pos)
            for neighbor in self.neighbors(cur_node):
                if self.accessible(neighbor) == False or neighbor.pos in close_set:
                    continue

                neighbor.cost = cur_node.cost + 1
                neighbor.parent = cur_node
                open_set.push(neighbor)  # 自动处理替换逻辑

        print("*****NO PATH!!!*****")
        return []


if __name__ == "__main__":
    dem = DEMGenerator((100,100),
                       (15, 15, 30, 10),
                       (30, 50, 100, 25),
                       (80, 80, 80, 15),
                       (80, 40, 60, 20))
    map = dem.gemDEM()
    dem.plt_DEM()

    astar = Astar(map, (6,20,50), (40,90,100))

    path = astar.search(True)
    # print(path)



