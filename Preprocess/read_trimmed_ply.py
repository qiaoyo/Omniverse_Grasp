import open3d as o3d
import os,sys
import numpy as np

data='/home/pika/LabelFusion_Sample_Data/logs/1/trimmed_log.lcmlog.ply'
ply=o3d.io.read_point_cloud(data)
points=np.asarray(ply.points)
print(points.shape)
