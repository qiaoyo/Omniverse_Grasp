import os

import open3d as o3d
import numpy as np


data_path='/home/pika/assemble_scale_grasp_001/'

data_save_path='/home/pika/object-grasp-annotation/models/'
parts_name=os.listdir(data_path)
parts_name=sorted(parts_name)
cnt=0
for part in parts_name:

    part_path=os.path.join(data_path,part)
    part_save_path=os.path.join(data_save_path,part)

    if not os.path.exists(part_save_path):
        os.makedirs(part_save_path)

    obj_path=os.path.join(part_path,part+'.obj')
    obj_target_path=os.path.join(part_save_path,'textured.obj')

    sdf_path='/home/pika/models/'+part+'/'+'textured.sdf'
    sdf_target_path=os.path.join(part_save_path,'textured.sdf')

    ply_target_path=os.path.join(part_save_path,'nontextured.ply')

    mesh = o3d.io.read_triangle_mesh(obj_path)

    raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 3)
    # print(np.array(raw_pc.points).shape,end=' ')
    voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
    # print(np.array(voxel_pc.points).shape)

    sample_voxel_size=0.001
    sample_pc = voxel_pc.voxel_down_sample(sample_voxel_size)
    while (np.array(sample_pc.points).shape[0]>(0.011*(np.array(voxel_pc.points).shape[0]))):
        sample_voxel_size = sample_voxel_size+ 0.001
        sample_pc = voxel_pc.voxel_down_sample(sample_voxel_size)

    sample_voxel_size=round(sample_voxel_size,3)
    print(np.array(sample_pc.points).shape[0], np.array(voxel_pc.points).shape[0], end=' ')
    print('\'' + part + '\':' + '(' + str(sample_voxel_size) + ', ' + str(0.001) + ', ' + str(10000) + '),')
    cnt += 1

    # for sample_voxel_size in [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.020]:
    #     sample_pc=voxel_pc.voxel_down_sample(sample_voxel_size)
    #
    #     if(np.array(sample_pc.points).shape[0]<(0.011*(np.array(voxel_pc.points).shape[0]))):
    #         # print(np.array(sample_pc.points).shape[0],np.array(voxel_pc.points).shape[0],end=' ')
    #         print('\''+part+'\':'+'('+str(sample_voxel_size)+', '+str(0.001)+', '+str(10000)+'),')
    #         cnt+=1
    #         break


    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.io.write_point_cloud(ply_target_path, voxel_pc)
    # o3d.visualization.draw_geometries([voxel_pc]+[axis_pcd])
    os.system('cp '+obj_path+' '+obj_target_path)
    os.system('cp '+sdf_path+' '+sdf_target_path)
print(cnt,len(parts_name))