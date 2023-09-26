import numpy as np
import os,sys
from autolab_core import RigidTransform
from quarternion import *
import open3d as o3d
np.set_printoptions(precision=4,suppress=True)

data_path='/media/pika/Joyoyo/temp/1/'
# data_path='/media/pika/Joyoyo/omniverse_assemble/001_1/'

# camera_pos shape: [rep_frame, camera_num, 3], camera_rot_im shape: [rep_frame, camera_num, 3], camera_rot_real shape: [rep_frame, camera_num]
# part_pos shape: [part_idx, 3], part_rot_im shape: [part_idx, 3], part_rot_real shape: [part_idx, 3]
camera_pos=np.load(data_path+'Camera_Pos.npy')
camera_rot_real=np.load(data_path+'Camera_Rot_rel.npy')
camera_rot_im=np.load(data_path+'Camera_Rot_im.npy')

part_pos=np.load(data_path+'Parts_Pos.npy')
part_rot_real=np.load(data_path+'Parts_Rot_rel.npy')
part_rot_im=np.load(data_path+'Parts_Rot_im.npy')
part_list=np.loadtxt(data_path+'Total_Parts.txt',dtype=str)

for i in range(camera_pos.shape[1]):
    camera_pos_temp=camera_pos[0][i]
    camera_rot_quat=np.empty(4)
    camera_rot_quat[0:3]=camera_rot_im[0][i]
    camera_rot_quat[3]=camera_rot_real[0][i]
    print("camera_"+str(i)+':',camera_pos_temp,camera_rot_quat,quaternion2euler(camera_rot_quat))

for i in range(len(part_pos)):
    part_rot_quat=np.empty(4)
    part_rot_quat[0:3]=part_rot_im[i]
    part_rot_quat[3]=part_rot_real[i]
    print(part_list[i],part_rot_quat,quaternion2euler(part_rot_quat))


point_cloud=np.load(data_path+'RenderProduct_omni_kit_widget_viewport_ViewportTexture_0/pointcloud/'+'pointcloud_0001.npy')
point_cloud=point_cloud/100

point_cloud_normals=np.load(data_path+'RenderProduct_omni_kit_widget_viewport_ViewportTexture_0/pointcloud/'+'pointcloud_normals_0001.npy')
point_cloud_rgb=np.load(data_path+'RenderProduct_omni_kit_widget_viewport_ViewportTexture_0/pointcloud/'+'pointcloud_rgb_0001.npy')
point_cloud_semantic=np.load(data_path+'RenderProduct_omni_kit_widget_viewport_ViewportTexture_0/pointcloud/'+'pointcloud_semantic_0001.npy')
axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

print("x:", "min:", np.min(point_cloud[:, 0]), "max:", np.max(point_cloud[:, 0]), "mean:", np.mean(point_cloud[:, 0]))
print("y:", "min:", np.min(point_cloud[:, 1]), "max:", np.max(point_cloud[:, 1]), "mean:", np.mean(point_cloud[:, 1]))
print("z:", "min:", np.min(point_cloud[:, 2]), "max:", np.max(point_cloud[:, 2]), "mean:", np.mean(point_cloud[:, 2]))

mesh_name='001_1'
mesh_file = '/home/pika/assemble_scale_grasp_001/' + mesh_name + '/' + mesh_name + '.obj'
mesh = o3d.io.read_triangle_mesh(mesh_file)

raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
pc = np.asarray(voxel_pc.points).T

idx=np.where(part_list==mesh_name)[0][0]

part_rot_quat=np.empty(4)
part_rot_quat[0:3]=part_rot_im[idx]
part_rot_quat[3]=part_rot_real[idx]

pc=np.dot(quat2rot(part_rot_quat),pc).T+part_pos[idx]/100
raw_pcd = o3d.geometry.PointCloud()
raw_pcd.points = o3d.utility.Vector3dVector(pc)
img_array=np.zeros((pc.shape[0],3))
img_array[:,0]=1
raw_pcd.colors=o3d.utility.Vector3dVector(img_array)

print(point_cloud.shape)
pc=o3d.geometry.PointCloud()
pc.points=o3d.utility.Vector3dVector(point_cloud)

camera_pcd=o3d.io.read_point_cloud(data_path+'0001.ply')
img=camera_pcd.colors
img=np.array(img)
img=img[::-1]
temp=np.array(camera_pcd.points)
print("x:", "min:", np.min(temp[:, 0]), "max:", np.max(temp[:, 0]), "mean:", np.mean(temp[:, 0]))
print("y:", "min:", np.min(temp[:, 1]), "max:", np.max(temp[:, 1]), "mean:", np.mean(temp[:, 1]))
print("z:", "min:", np.min(temp[:, 2]), "max:", np.max(temp[:, 2]), "mean:", np.mean(temp[:, 2]))

camera_rot_quat=np.empty(4)
camera_rot_quat[0:3]=camera_rot_im[0][0]
camera_rot_quat[3]=camera_rot_real[0][0]
camera2world_T=RigidTransform(quat2rot(camera_rot_quat),camera_pos[0][0]/100)
print(camera2world_T)
print("camera_euler:",quaternion2euler(camera_rot_quat))
world2camera=camera2world_T.inverse().matrix
# camera_point_cloud=np.dot(world2camera[0:3,0:3],(np.array(camera_pcd.points)-camera_pos[0][0]/100).T).T
# camera_point_cloud=np.dot(world2camera[0:3,0:3],temp.T).T+world2camera[0:3,3].reshape(-1,3)
camera_point_cloud=np.dot(np.eye(3),temp.T).T+world2camera[0:3,3].reshape(-1,3)

camera_point_cloud[:,2]=-camera_point_cloud[:,2]

rot=np.array([  -5.4676,   0.   ,  -37.2519])
rot_matrix=euler2rot(rot)
camera_point_cloud=np.dot(rot_matrix,camera_point_cloud.T).T

print(camera_pos[0][0])
theta=math.atan2(camera_pos[0][0][0],camera_pos[0][0][1])*180/np.pi
print('theta:',theta)
rot=np.array([0,0,theta])
rot_matrix=euler2rot(rot)
camera_point_cloud=np.dot(rot_matrix,camera_point_cloud.T).T
camera_point_cloud[:,1]=-camera_point_cloud[:,1]

original_scene_pc=np.array(pc.points)
original_scene_pc=np.dot(rot_matrix,original_scene_pc.T).T



# camera_point_cloud[:,2]=-camera_point_cloud[:,2]

print("x:", "min:", np.min(camera_point_cloud[:, 0]), "max:", np.max(camera_point_cloud[:, 0]), "mean:", np.mean(camera_point_cloud[:, 0]))
print("y:", "min:", np.min(camera_point_cloud[:, 1]), "max:", np.max(camera_point_cloud[:, 1]), "mean:", np.mean(camera_point_cloud[:, 1]))
print("z:", "min:", np.min(camera_point_cloud[:, 2]), "max:", np.max(camera_point_cloud[:, 2]), "mean:", np.mean(camera_point_cloud[:, 2]))

print('world2camera:',world2camera)
origin=np.dot(world2camera[0:3,0:3],np.array([-0.4847, -1.095,  2.8906]))+world2camera[0:3,3]
print('orgin:',origin)

camera_pcd.points=o3d.utility.Vector3dVector(camera_point_cloud)
# camera_pcd.colors=o3d.utility.Vector3dVector(img)
point_cloud=np.dot(quat2rot(camera_rot_quat),point_cloud.T).T+camera_pos[0][0]/100
# pc.points=o3d.utility.Vector3dVector(point_cloud)
pc.points=o3d.utility.Vector3dVector(original_scene_pc)

o3d.visualization.draw_geometries([pc]+[axis_pcd]+[raw_pcd]+[camera_pcd])