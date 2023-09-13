import open3d
import numpy as np
import mayavi.mlab as mlab

# 对两个ply格式的点云数据进行配准
def open3d_registration():
    ply_path = "/home/pika/LabelFusion_Sample_Data/logs/0904_top/trimmed_log.lcmlog.ply"
    ply_1 = open3d.io.read_triangle_mesh(ply_path)
    point_s1 = np.array(ply_1.vertices)

    ply_path = "/home/pika/LabelFusion_Sample_Data/logs/0904_right/trimmed_log.lcmlog.ply"
    ply_2 = open3d.io.read_triangle_mesh(ply_path)
    point_s2 = np.array(ply_2.vertices)

    print(point_s1.shape, point_s2.shape)

    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(point_s1)
    target = open3d.geometry.PointCloud()
    target.points = open3d.utility.Vector3dVector(point_s2)

    icp = open3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=0.2,    # 距离阈值
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(icp)

    source.transform(icp.transformation)
    points = np.array(source.points)

    # 配准可视化
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
    mlab.points3d(x, y, z, x, mode="point",  figure=fig)   # 蓝色

    # x = point_s1[:, 0]
    # y = point_s1[:, 1]
    # z = point_s1[:, 2]
    # mlab.points3d(x, y, z, x, mode="point", color=(0, 1, 0), figure=fig)   # 绿色
    #
    # x = point_s2[:, 0]
    # y = point_s2[:, 1]
    # z = point_s2[:, 2]
    # mlab.points3d(x, y, z, x, mode="point", color=(1, 0, 0), figure=fig)   # 红色
    mlab.show()

    # 拼接可视化
    # points = np.concatenate([points, point_s2], axis=0)
    # print(points.shape)

    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
    # mlab.points3d(x, y, z, y, mode="point", figure=fig)
    # mlab.show()


if __name__ == '__main__':
    open3d_registration()
