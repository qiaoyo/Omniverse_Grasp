import numpy as np
import os,sys
from autolab_core import RigidTransform
from quarternion import *
import open3d as o3d
import yaml
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper,GpgGraspSampler
np.set_printoptions(precision=4,suppress=True)
yaml_config = YamlConfig("/home/pika/Omniverse_Grasp/dex-net/test/config.yaml")
gripper = RobotGripper.load('robotiq_85', gripper_dir="../dex-net/data/grippers")
GS=GpgGraspSampler(gripper,yaml_config)

def scene2camera(scene_point_cloud,camera2world,world2camera):

    camera2world_roteuler = rot2euler(camera2world[0:3, 0:3])
    world2camera_roteuler = rot2euler(world2camera[0:3, 0:3])
    print('camera2world: rot_euler:', camera2world_roteuler)
    print('world2camera: rot_euler:', world2camera_roteuler)

    rot_matrix = euler2rot([0, 0, -camera2world_roteuler[2]])
    scene_point_cloud = np.dot(rot_matrix, scene_point_cloud.T).T

    scene_point_cloud[:, 2] = -scene_point_cloud[:, 2]
    scene_point_cloud[:, 1] = -scene_point_cloud[:, 1]

    rot_matrix = euler2rot([-camera2world_roteuler[0], 0, 0])
    scene_point_cloud = np.dot(rot_matrix, scene_point_cloud.T).T

    scene_point_cloud = np.dot(np.eye(3), scene_point_cloud.T).T - world2camera[0:3, 3].reshape(-1, 3)

    return scene_point_cloud

def mesh2scene(mesh_pc,mesh2world):
    mesh_pc=np.dot(mesh2world[0:3,0:3],mesh_pc.T).T+mesh2world[0:3,3]

    return mesh_pc

def scene2mesh(scene_pc,world2mesh):
    scene_pc = np.dot(world2mesh[0:3,0:3], scene_pc.T).T + world2mesh[0:3,3]
    return scene_pc

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def npy2geometry(grasp,vertices,colors=None):
    '''
    Description: convert the grasp npy and the related 32 gripper points to the TriangleMesh format to visualize the grippers in the camera_point_cloud.
    :param grasp: shape: [17]
    :param vertices: shape: [32]
    :return:
    '''
    depth=grasp[3]
    width=grasp[1]
    score=grasp[0]
    color_r = score  # red for high score
    color_g = 0
    color_b = 1 - score  # blue for low score
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    if colors is None:
        colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper=o3d.geometry.TriangleMesh()
    gripper.vertices=o3d.utility.Vector3dVector(vertices)
    gripper.triangles=o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors=o3d.utility.Vector3dVector(colors)
    return gripper

def gripper2scene(grippers,mesh_name,camera2world,world2camera,mesh2world):
    grippers_new=[]
    for i,gripper in enumerate(grippers):
        vertices=gripper.vertices
        vertices=np.array(vertices)
        points_vertices=o3d.geometry.PointCloud()
        print('gripper:',vertices.shape)
        vertices=mesh2scene(vertices,mesh2world)
        vertices=scene2camera(vertices,camera2world,world2camera)
        if i  ==0:
            # no collision
            colors = np.array([[0, 1, 0] for _ in range(len(vertices))])
        else:
            # collision
            colors=np.array([[1,0,0] for _ in range(len(vertices))])

        gripper_new=npy2geometry(grasp_group[i],vertices,colors)
        grippers_new.append(gripper_new)

    return grippers_new

def get_world_camera_transform(camera_rot_im,camera_rot_real,camera_pos,camera_idx,random_idx,render_steps):
    if random_idx==0:
        random_idx=render_steps-1
    else:
        random_idx=random_idx-1
    camera_rot_quat = np.empty(4)
    camera_rot_quat[0:3] = camera_rot_im[random_idx][camera_idx]
    camera_rot_quat[3] = camera_rot_real[random_idx][camera_idx]
    camera2world_T = RigidTransform(quat2rot(camera_rot_quat), camera_pos[random_idx][camera_idx])
    # print("camera2world:", camera2world_T)
    camera2world = camera2world_T.matrix
    world2camera_T = camera2world_T.inverse()
    # print("world2camera:", world2camera_T)
    world2camera = world2camera_T.matrix

    return camera2world,world2camera

def get_mesh_world_transform(mesh_name,part_list,part_pos,part_rot_im,part_rot_real):
    mesh_idx = np.where(part_list == mesh_name)[0][0]

    part_rot_quat = np.empty(4)
    part_rot_quat[0:3] = part_rot_im[mesh_idx]
    part_rot_quat[3] = part_rot_real[mesh_idx]

    mesh2world_T=RigidTransform(quat2rot(part_rot_quat),part_pos[mesh_idx])
    mesh2world=mesh2world_T.matrix
    # print("mesh2world:",mesh2world_T)
    world2mesh_T=mesh2world_T.inverse()
    # print("world2mesh:",world2mesh_T)
    world2mesh=world2mesh_T.matrix

    return mesh2world,world2mesh

if __name__=='__main__':
    with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.0/Omniverse_Grasp/config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)
    simulation_steps=Config_yaml['Renderer']['Simulation_steps']
    render_steps=Config_yaml['Renderer']['Render_steps']

    camera_idx = 0
    random_idx = 0 # 0~9
    rgbd_idx = str(random_idx + 1).rjust(4, '0') # 0001~0010
    camera_name_list = ['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0', 'RenderProduct_Replicator',
                        'RenderProduct_Replicator_01', 'RenderProduct_Replicator_02', 'RenderProduct_Replicator_03',
                        'RenderProduct_Replicator_04', 'RenderProduct_Replicator_05', 'RenderProduct_Replicator_06']

    data_path = '/media/pika/Joyoyo/temp/1/'
    img_path = data_path + camera_name_list[camera_idx] + '/rgb/'
    depth_path = data_path + camera_name_list[camera_idx] + '/distance_to_camera/'
    mesh_path='/home/pika/assemble_scale_grasp_001/'

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

    camera2world,world2camera=get_world_camera_transform(camera_rot_im,camera_rot_real,camera_pos,camera_idx,random_idx,render_steps)
    mesh2world,world2mesh=get_mesh_world_transform(mesh_name,part_list,part_pos,part_rot_im,part_rot_real)

    mesh_file = mesh_path + mesh_name + '/' + mesh_name + '.obj'
    mesh_obj = o3d.io.read_triangle_mesh(mesh_file)

    raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_obj, np.asarray(mesh_obj.vertices).shape[0] * 10)
    mesh_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
    mesh_pc = np.asarray(mesh_pc.points)


    mesh_pcd=o3d.geometry.PointCloud()
    scene_pcd = o3d.geometry.PointCloud()
    camera_pcd = o3d.io.read_point_cloud(depth_path + rgbd_idx + '.ply')

    camera_point_cloud = np.array(camera_pcd.points)

    img_array=np.zeros((mesh_pc.shape))
    img_array[:,2]=1
    mesh_pcd.colors=o3d.utility.Vector3dVector(img_array)

    point_cloud = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_'+rgbd_idx+'.npy')/100
    point_cloud_normals = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_normals_'+rgbd_idx+'.npy')/100
    point_cloud_rgb = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_rgb_'+rgbd_idx+'.npy')/100
    point_cloud_semantic = np.load(
        data_path + camera_name_list[camera_idx]+'/pointcloud/' + 'pointcloud_semantic_'+rgbd_idx+'.npy')/100


    scene_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    scene_point_cloud = np.array(scene_pcd.points)

    scene_point_cloud=scene2camera(scene_point_cloud,camera2world,world2camera)
    mesh_pc = mesh2scene(mesh_pc, mesh2world)
    mesh_pc=scene2camera(mesh_pc,camera2world,world2camera)

    # camera_point_cloud=np.dot(np.eye(3),camera_point_cloud.T).T+world2camera[0:3,3].reshape(-1,3)

    axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=-world2camera[0:3,3])
    # axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])

    scene_pcd.points=o3d.utility.Vector3dVector(scene_point_cloud)
    camera_pcd.points=o3d.utility.Vector3dVector(camera_point_cloud)
    mesh_pcd.points=o3d.utility.Vector3dVector(mesh_pc)

    from visualize_obj_grasplabel import visualizeGrasp
    grasp_group,grippers=visualizeGrasp(mesh_name,num_sample=10,random=False)
    print(grasp_group)
    grippers_new=[]
    grippers_new=gripper2scene(grippers,mesh_name,camera2world,world2camera,mesh2world)

    # check the 32 gripper points
    # vertices = gripper.vertices
    # vertices = np.array(vertices)
    # points_vertices = o3d.geometry.PointCloud()
    # tt=0
    # points_vertices.points=o3d.utility.Vector3dVector([vertices[tt],vertices[tt+1],vertices[tt+2],vertices[tt+3]])
    # temp_color=np.zeros((4,3))
    # temp_color[0]=np.array([1,0,0])
    # temp_color[1]=np.array([0,1,0])
    # temp_color[2]=np.array([0,0,1])
    # temp_color[3]=np.array([1,1,1])
    # points_vertices.colors=o3d.utility.Vector3dVector(temp_color)
    # o3d.visualization.draw_geometries([axis_pcd] + [points_vertices] + gripper)


    # x:red, y:green. z:blue

    o3d.visualization.draw_geometries([camera_pcd]+[]+grippers_new+[mesh_pcd]+[axis_pcd])
