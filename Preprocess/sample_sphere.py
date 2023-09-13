import numpy as np
import open3d as o3d
# 生成球面均匀分布的点
n = 256
theta = np.linspace(0, np.pi/2, int(np.sqrt(n) * 2))
phi = np.linspace(0, 2 * np.pi, int(np.sqrt(n) * 2))
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

# 转换球坐标为xyz坐标
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# 打印前几个点的坐标
for i in range(10):
    print(f"Point {i+1}: ({x[i]:.6f}, {y[i]:.6f}, {z[i]:.6f})")
# 最终的xyz坐标在 x、y 和 z 数组中


def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
        idxs = np.arange(N, dtype=np.float32)
        Z = ( idxs + 1) / N
        X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
        Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
        views = np.stack([X,Y,Z], axis=1)
        views = R * np.array(views) + center
        return  views

views = generate_views(N=256)
print(views.shape)

# xyz_points = np.column_stack((x, y, z))
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(views)

axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])
# 创建绘制窗口
o3d.visualization.draw_geometries([point_cloud]+[axis_pcd])