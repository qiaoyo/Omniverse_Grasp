import numpy as np
import os,sys
from autolab_core import RigidTransform
from quarternion import *
import open3d as o3d


axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

right_view=np.array( [-0.3376646696333002, -0.05100443535190555, 0.8024664754156701, 0.6690162078981932, -0.6610694153388643, 0.25322175969820376, 0.22645700548323916])
top_view=np.array( [-0.19441574775204462, -0.05942858684099318, 0.6830320979182918, 0.7006939478139234, -0.6746128569395263, 0.157837325856877, -0.17026933292358026])

right_view_trans=right_view[0:3]
right_view_rot_quan=right_view[3:]
top_view_trans=top_view[0:3]
top_view_rot_quan=top_view[3:]

pcd_top=o3d.io.read_point_cloud('/home/pika/LabelFusion_Sample_Data/logs/0904_top/trimmed_log.lcmlog.ply')
pcd_right=o3d.io.read_point_cloud('/home/pika/LabelFusion_Sample_Data/logs/0904_right/trimmed_log.lcmlog.ply')

pcd_right_points=np.asarray(pcd_right.points)
pcd_top_points=np.asarray(pcd_top.points)
print(pcd_right_points.shape)

april2camera_top=RigidTransform(quat2rot(top_view_rot_quan),top_view_trans)
april2camera_right=RigidTransform(quat2rot(right_view_rot_quan),right_view_trans)

camera_top2april=april2camera_top.inverse()
camera_top2april=camera_top2april.matrix
april2camera_right=april2camera_right.matrix


top2right=np.dot(camera_top2april,april2camera_right)
pcd_right_points=np.dot(top2right[0:3,0:3],pcd_right_points.T).T+top2right[0:3,3].reshape(1,3)
pcd_right_new=o3d.geometry.PointCloud()
pcd_right_new.points=o3d.utility.Vector3dVector(pcd_top_points)
o3d.visualization.draw_geometries([]+[pcd_right]+[pcd_right_new]+[axis_pcd])

#
# right2top=np.dot(april2camera_right,camera_top2april)
#
# pcd_top_points=np.dot(right2top[0:3,0:3],pcd_top_points.T).T+right2top[0:3,3].reshape(1,3)
#
# pcd_top_new=o3d.geometry.PointCloud()
# pcd_top_new.points=o3d.utility.Vector3dVector(pcd_top_points)
#
# o3d.visualization.draw_geometries([pcd_top]+[pcd_right]+[pcd_top_new]+[axis_pcd])
