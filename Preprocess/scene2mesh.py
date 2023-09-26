import time
import numpy as np
import os,sys
import open3d as o3d
import yaml
from quarternion import *
from visualize import get_mesh_world_transform,get_world_camera_transform,scene2mesh,scene2camera
from collision_check import graspnpy2graspgroup
from collision_check import ModelFreeCollisionDetector
from multiprocessing import Process
def camera2mesh(camera_point_cloud,camera2world,world2camera,mesh2world,world2mesh):

    camera_point_cloud=np.dot(np.eye(3),camera_point_cloud.T).T+world2camera[0:3,3].reshape(-1,3)

    camera2world_euler=rot2euler(camera2world[0:3,0:3])
    rot_euler=np.array( [-camera2world_euler[0],    0., 0])
    rot_matrix=np.linalg.inv(euler2rot(rot_euler))
    camera_point_cloud=np.dot(rot_matrix,camera_point_cloud.T).T
    camera_point_cloud[:,2]=-camera_point_cloud[:,2]
    camera_point_cloud[:,1]=-camera_point_cloud[:,1]
    rot_euler=np.array( [0, 0., -camera2world_euler[2]])
    rot_matrix=np.linalg.inv(euler2rot(rot_euler))
    camera_point_cloud=np.dot(rot_matrix,camera_point_cloud.T).T

    camera_point_cloud=scene2mesh(camera_point_cloud,world2mesh)

    return camera_point_cloud

def worker(part_name):
    for j in range(1,11):
        data_path = '/media/pika/Joyoyo/0921/'+part_name+'_'+str(j)+'/'
        grasp_path='/home/pika/good_grasp/'
        camera_name_list = ['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0', 'RenderProduct_Replicator',
                            'RenderProduct_Replicator_01', 'RenderProduct_Replicator_02', 'RenderProduct_Replicator_03',
                            'RenderProduct_Replicator_04', 'RenderProduct_Replicator_05', 'RenderProduct_Replicator_06']
        with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.0/Omniverse_Grasp/config.yaml') as Config_file:
            Config_yaml = yaml.load(Config_file, Loader=yaml.FullLoader)
        simulation_steps = Config_yaml['Renderer']['Simulation_steps']
        render_steps = Config_yaml['Renderer']['Render_steps']
        camera_num=Config_yaml['Camera']['num']
        mesh_path = '/home/pika/assemble_scale_grasp_001/'
        part_list = np.loadtxt(data_path + 'Total_Parts.txt', dtype=str)
        grasp_npy_path = '/home/pika/object-grasp-annotation/grasp_label/'
        show_mesh_pcd=False
        save_scene_pcd=False

        camera_pos = np.load(data_path + 'Camera_Pos.npy')
        camera_pos = camera_pos / 100
        camera_rot_real = np.load(data_path + 'Camera_Rot_rel.npy')
        camera_rot_im = np.load(data_path + 'Camera_Rot_im.npy')

        part_pos = np.load(data_path + 'Parts_Pos.npy')
        part_pos = part_pos / 100
        part_rot_real = np.load(data_path + 'Parts_Rot_rel.npy')
        part_rot_im = np.load(data_path + 'Parts_Rot_im.npy')

        t_start=time.time()

        for mesh_name in part_list:
            mesh_start=time.time()
            flag=False
            # mesh_collision=np.zeros((num_grasp),dtype=bool)
            mesh_collision = np.load(data_path + mesh_name + '_collision.npy')
            grasp_label = np.load(grasp_npy_path + mesh_name + '_labels.npz')
            grasp_Group = graspnpy2graspgroup(grasp_label)
            num_grasp = len(grasp_Group.grasp_group_array)

            for random_idx in range(render_steps):
                if flag==True:
                    break
                for camera_idx in range(1,camera_num):
                    if flag==True:
                        break
                    rgbd_idx = str(random_idx + 1).rjust(4, '0')  # 0001~0010
                    img_path = data_path + camera_name_list[camera_idx] + '/rgb/'
                    depth_path = data_path + camera_name_list[camera_idx] + '/distance_to_camera/'

                    # camera_pos shape: [rep_frame, camera_num, 3], camera_rot_im shape: [rep_frame, camera_num, 3], camera_rot_real shape: [rep_frame, camera_num]
                    # part_pos shape: [part_idx, 3], part_rot_im shape: [part_idx, 3], part_rot_real shape: [part_idx, 3]

                    mesh_idx = np.where(part_list == mesh_name)[0][0]

                    if show_mesh_pcd:
                        mesh_file = '/home/pika/assemble_scale_grasp_001/' + mesh_name + '/' + mesh_name + '.obj'
                        mesh = o3d.io.read_triangle_mesh(mesh_file)

                        raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
                        voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
                        mesh_pc = np.asarray(voxel_pc.points)
                        mesh_pcd=o3d.geometry.PointCloud()
                        mesh_pcd.points=o3d.utility.Vector3dVector(mesh_pc)

                    camera_pcd = o3d.io.read_point_cloud(depth_path + rgbd_idx + '.ply')
                    camera_point_cloud = np.array(camera_pcd.points)

                    camera2world,world2camera=get_world_camera_transform(camera_rot_im,camera_rot_real,camera_pos,camera_idx,random_idx,render_steps)
                    mesh2world,world2mesh=get_mesh_world_transform(mesh_name,part_list,part_pos,part_rot_im,part_rot_real)
                    camera_point_cloud = camera2mesh(camera_point_cloud,camera2world,world2camera,mesh2world,world2mesh)

                    mfcDetector = ModelFreeCollisionDetector(camera_point_cloud)
                    collision_mask = mfcDetector.detect(grasp_Group, collision=mesh_collision,approach_dist=0.03)

                    origninal_num=np.sum(mesh_collision==False)
                    mesh_collision=np.logical_or(mesh_collision,collision_mask)
                    checked_num=np.sum(mesh_collision==False)

                    print(part_name,mesh_name,' in render: ',random_idx," in camera: ",camera_idx," from ",origninal_num,' to ',checked_num)
                    if checked_num==0:
                        flag=True
                    if save_scene_pcd:
                        o3d.io.write_point_cloud(depth_path+rgbd_idx+'_'+mesh_name+'.ply',camera_pcd)
                    # print(depth_path+rgbd_idx+'_'+mesh_name+'.ply'+' done!')
                    if show_mesh_pcd:
                        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        camera_pcd.points = o3d.utility.Vector3dVector(camera_point_cloud)
                        o3d.visualization.draw_geometries([camera_pcd]+[axis_pcd]+[mesh_pcd])
                    # break
            mesh_end=time.time()
            np.save(data_path+mesh_name+'_collision_new.npy',mesh_collision)
            print(part_name+'_'+str(j)+': '+mesh_name+':',mesh_end-mesh_start,'s')
        t_end=time.time()

        print('total_time: ',t_end-t_start,'s')

def main():
    PART=os.listdir('/home/pika/assemble_step/')
    pool=[]
    for part_name in PART:
        print(part_name)
        task=Process(target=worker,args=[part_name])
        task.start()
        pool.append(task)
    for task in pool:
        task.join()


if __name__=='__main__':
    main()
   # worker('001')
