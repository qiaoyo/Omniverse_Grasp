import numpy as np
import open3d as o3d
import os,sys

if __name__=='__main__':
    data_path='/home/pika/LabelFusion_Sample_Data/resample_parts/074_step_OBJ/074.obj'

    mesh=o3d.io.read_triangle_mesh(data_path)
    print(mesh.vertices)
    raw_pc=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,np.asarray(mesh.vertices).shape[0]*5)

    voxel_pc=o3d.geometry.PointCloud.voxel_down_sample(raw_pc,0.001)

    o3d.io.write_point_cloud('/home/pika/LabelFusion_Sample_Data/resample_parts/074_step_OBJ/074.vtp')

    o3d.visualization.draw_geometries([voxel_pc])