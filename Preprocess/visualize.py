import numpy as np
import os,sys
from autolab_core import RigidTransform
from quarternion import *
import open3d as o3d
np.set_printoptions(precision=4,suppress=True)

def scene2camera(scene_point_cloud,camera_rot_im,camera_rot_real,camera_pos,camera_idx,random_idx):
    random_idx=5
    camera_rot_quat = np.empty(4)
    camera_rot_quat[0:3] = camera_rot_im[random_idx][camera_idx]
    camera_rot_quat[3] = camera_rot_real[random_idx][camera_idx]
    camera2world_T = RigidTransform(quat2rot(camera_rot_quat), camera_pos[random_idx][camera_idx])
    print("camera2world:", camera2world_T)
    camera2world = camera2world_T.matrix
    world2camera_T = camera2world_T.inverse()
    world2camera = world2camera_T.matrix
    print("world2camera:", world2camera)

    camera2world_roteuler = rot2euler(camera2world[0:3, 0:3])
    world2camera_roteuler = rot2euler(world2camera[0:3, 0:3])
    print('rot_euler:', camera2world_roteuler)
    print('rot_euler:', world2camera_roteuler)

    rot_matrix = euler2rot([0, 0, -camera2world_roteuler[2]])
    scene_point_cloud = np.dot(rot_matrix, scene_point_cloud.T).T

    scene_point_cloud[:, 2] = -scene_point_cloud[:, 2]
    scene_point_cloud[:, 1] = -scene_point_cloud[:, 1]

    rot_matrix = euler2rot([-camera2world_roteuler[0], 0, 0])
    scene_point_cloud = np.dot(rot_matrix, scene_point_cloud.T).T

    scene_point_cloud = np.dot(np.eye(3), scene_point_cloud.T).T - world2camera[0:3, 3].reshape(-1, 3)

    return scene_point_cloud,world2camera,camera2world

def mesh2scene(mesh_pc,mesh_name,part_list,part_pos,part_rot_im,part_rot_real):
    mesh_idx=np.where(part_list==mesh_name)[0][0]


    part_rot_quat=np.empty(4)
    part_rot_quat[0:3]=part_rot_im[mesh_idx]
    part_rot_quat[3]=part_rot_real[mesh_idx]

    mesh_pc=np.dot(quat2rot(part_rot_quat),mesh_pc.T).T+part_pos[mesh_idx]

    return mesh_pc

if __name__=='__main__':
    camera_idx = 4
    random_idx = 0 # 0~9
    rgbd_idx = str(random_idx + 1).rjust(4, '0') # 0001~0010
    camera_name_list = ['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0', 'RenderProduct_Replicator',
                        'RenderProduct_Replicator_01', 'RenderProduct_Replicator_02', 'RenderProduct_Replicator_03',
                        'RenderProduct_Replicator_04', 'RenderProduct_Replicator_05', 'RenderProduct_Replicator_06']

    data_path = '/media/pika/Joyoyo/temp/1/'
    img_path = data_path + camera_name_list[camera_idx] + '/rgb/'
    depth_path = data_path + camera_name_list[camera_idx] + '/distance_to_camera/'

    part_list = np.loadtxt(data_path + 'Total_Parts.txt', dtype=str)

    # camera_pos shape: [rep_frame, camera_num, 3], camera_rot_im shape: [rep_frame, camera_num, 3], camera_rot_real shape: [rep_frame, camera_num]
    # part_pos shape: [part_idx, 3], part_rot_im shape: [part_idx, 3], part_rot_real shape: [part_idx, 3]
    camera_pos = np.load(data_path + 'Camera_Pos.npy')
    camera_pos = camera_pos / 100
    camera_rot_real = np.load(data_path + 'Camera_Rot_rel.npy')
    camera_rot_im = np.load(data_path + 'Camera_Rot_im.npy')

    part_pos = np.load(data_path + 'Parts_Pos.npy')
    part_pos = part_pos / 100
    part_rot_real = np.load(data_path + 'Parts_Rot_rel.npy')
    part_rot_im = np.load(data_path + 'Parts_Rot_im.npy')

    mesh_name='001_1'
    mesh_idx=np.where(part_list==mesh_name)[0][0]
    mesh_file = '/home/pika/assemble_scale_grasp_001/' + mesh_name + '/' + mesh_name + '.obj'
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
    voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
    mesh_pc = np.asarray(voxel_pc.points)
    mesh_pc = mesh2scene(mesh_pc, mesh_name, part_list, part_pos, part_rot_im, part_rot_real)

    mesh_pcd=o3d.geometry.PointCloud()
    img_array=np.zeros((mesh_pc.shape))
    img_array[:,0]=1
    mesh_pcd.colors=o3d.utility.Vector3dVector(img_array)

    point_cloud = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_'+rgbd_idx+'.npy')
    point_cloud = point_cloud / 100
    point_cloud_normals = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_normals_'+rgbd_idx+'.npy')
    point_cloud_rgb = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_rgb_'+rgbd_idx+'.npy')
    point_cloud_semantic = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_semantic_'+rgbd_idx+'.npy')

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    scene_point_cloud = np.array(scene_pcd.points)

    scene_point_cloud,world2camera,camera2world=scene2camera(scene_point_cloud,camera_rot_im=camera_rot_im,camera_rot_real=camera_rot_real,camera_pos=camera_pos,camera_idx=camera_idx,random_idx=random_idx)
    mesh_pc,world2camera,camera2world=scene2camera(mesh_pc,camera_rot_im=camera_rot_im,camera_rot_real=camera_rot_real,camera_pos=camera_pos,camera_idx=camera_idx,random_idx=random_idx)

    camera_pcd = o3d.io.read_point_cloud(depth_path + rgbd_idx+'.ply')
    camera_point_cloud = np.array(camera_pcd.points)

    # camera_point_cloud=np.dot(np.eye(3),camera_point_cloud.T).T+world2camera[0:3,3].reshape(-1,3)

    axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=-world2camera[0:3,3])
    # axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

    scene_pcd.points=o3d.utility.Vector3dVector(scene_point_cloud)
    camera_pcd.points=o3d.utility.Vector3dVector(camera_point_cloud)
    mesh_pcd.points=o3d.utility.Vector3dVector(mesh_pc)

    o3d.visualization.draw_geometries([camera_pcd]+[scene_pcd]+[mesh_pcd]+[axis_pcd])
