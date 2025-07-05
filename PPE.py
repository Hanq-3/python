import numpy as np
import matplotlib.pyplot as plt


class genDEM:
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


if __name__ == "__main__":
    dem = genDEM((100,100),
                 (15, 15, 30, 10),
                 (30, 50, 100, 25),
                 (80, 80, 80, 15),
                 (80, 40, 60, 20))
    map = dem.gemDEM()

    dem.plt_DEM()



