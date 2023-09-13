import copy
import pickle
import numpy as np
import os,sys
import tqdm
import torch
from mayavi import mlab
from quarternion import *
import scipy
from autolab_core import RigidTransform,YamlConfig
import open3d as o3d
np.set_printoptions(suppress=True,precision=6)
from dexnet.grasping import RobotGripper,GpgGraspSampler
from PIL import Image
# to be modified
yaml_config = YamlConfig("/home/pika/Omniverse_Grasp/dex-net/test/config.yaml")
gripper = RobotGripper.load('robotiq_85', gripper_dir="../dex-net/data/grippers")
GS=GpgGraspSampler(gripper,yaml_config)

def get_rot_mat(poses_vector):
    '''从抓取向量中计算出夹爪相对于mesh模型的姿态
    '''
    major_pc = poses_vector[:, 3:6]  # (-1,3)
    angle = poses_vector[:, [7]]  # (-1,1)

    # cal approach
    cos_t = np.cos(angle)  # (-1,1)
    sin_t = np.sin(angle)
    zeros = np.zeros(cos_t.shape)  # (-1,1)
    ones = np.ones(cos_t.shape)

    # 绕抓取binormal轴的旋转矩阵
    R1 = np.c_[cos_t, zeros, -sin_t, zeros, ones, zeros, sin_t, zeros, cos_t].reshape(-1, 3, 3)  # [len(grasps),3,3]
    # print(R1)
    axis_y = major_pc  # (-1,3)

    # 设定一个与抓取y轴垂直且与C:x-o-y平面平行的单位向量作为初始x轴
    axis_x = np.c_[axis_y[:, [1]], -axis_y[:, [0]], zeros]
    # 查找模为0的行，替换为[1,0,0]
    axis_x[np.linalg.norm(axis_x, axis=1) == 0] = np.array([1, 0, 0])
    # 单位化
    axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
    # 右手定则，从x->y
    axis_z = np.cross(axis_x, axis_y)

    # 这个R2就是一个临时的夹爪坐标系，但是它的姿态还不代表真正的夹爪姿态
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]].reshape(-1, 3, 3).swapaxes(1, 2)
    # 将现有的坐标系利用angle进行旋转，就得到了真正的夹爪坐标系，
    # 抽出x轴作为approach轴(原生dex-net夹爪坐标系)
    # 由于是相对于运动坐标系的旋转，因此需要右乘
    R3 = np.matmul(R2, R1)
    '''
    approach_normal =R3[:, :,0]
    #print(np.linalg.norm(approach_normal,axis=1,keepdims=True))
    approach_normal = approach_normal / np.linalg.norm(approach_normal,axis=1,keepdims=True)
    #minor_pc=R3[:, :,2]  是一样的
    minor_pc = np.cross( approach_normal,major_pc)
    '''
    # 然后把平移向量放在每个旋转矩阵的最右侧，当成一列
    return R3

def get_scene_grasp(mesh_list,grasp_npy,camera_pos,camera_rot_im,camera_rot_real,part_pos,part_rot_im,part_rot_real):
    random_index=1
    camera_index=0
    hand_depth=0.125
    # world to camera
    camera_rot=np.empty([4])
    camera_rot[0]=camera_rot_real[random_index][camera_index]
    camera_rot[1:]=camera_rot_im[random_index][camera_index]
    world2camera_trans=camera_pos[random_index][camera_index]
    world2camera_rot=RigidTransform.rotation_from_quaternion(camera_rot)
    world2camera_T=RigidTransform(world2camera_rot,world2camera_trans)

    camera2world_T=world2camera_T.inverse().matrix

    # grasp to gripper bottom
    grasp2bottom_T=RigidTransform(np.eye(3),np.array([-hand_depth,0,0])).matrix

    grasp_center=np.empty(shape=(0,3))
    grasp_pose=np.empty(shape=(0,3,3))
    grasp_score=np.empty(shape=(0,1))
    hand_point=np.empty(shape=(0,21,3))

    #
    # grasp_center = grasp_npy[0][0:3]
    # grasp_axis = grasp_npy[0][3:6]
    # width = grasp_npy[0][6]
    # angle = grasp_npy[0][7]
    # jaw_width = grasp_npy[0][8]
    # min_width = grasp_npy[0][9]

    local_hand_point=GS.get_hand_points(np.array([0,0,0]),np.array([1,0,0]),np.array([0,1,0]))

    for i,mesh in enumerate(mesh_list):
        if mesh=='001_1':
            print(i)
            mesh_rot=np.empty([4])
            mesh_rot[0]=part_rot_real[i]
            mesh_rot[1:]=part_rot_im[i]
            world2mesh_T=RigidTransform(mesh_rot,part_pos[i]).matrix

            mesh2grasp_rot=get_rot_mat(grasp_npy)
            mesh2grasp_trans=grasp_npy[:,0:3]

            mesh2grasp_rot_trans=np.concatenate((mesh2grasp_rot,mesh2grasp_trans.reshape(-1,3,1)),axis=2)
            temp=np.array([0,0,0,1]).reshape(1,1,4).repeat(mesh2grasp_rot_trans.shape[0],axis=0)
            mesh2grasp_T=np.concatenate((mesh2grasp_rot_trans,temp),axis=1)

            # print(mesh2grasp_T.shape,mesh2grasp_T[0])

            camera2mesh_T=np.matmul(camera2world_T,world2mesh_T)
            camera2mesh_trans=camera2mesh_T[0:3,3]

            camera2grasp_T=np.matmul(np.matmul(camera2world_T,world2mesh_T),mesh2grasp_T)
            camera2grasp_rot=camera2grasp_T[:,0:3,0:3]
            camera2grasp_trans=camera2grasp_T[:,0:3,3].reshape(-1,3)

            camera2bottom_T=np.matmul(camera2grasp_T,grasp2bottom_T)
            camera2bottom_rot=camera2bottom_T[:,0:3,0:3]
            camera2bottom_trans=camera2bottom_T[:,0:3,3].reshape(-1,3)

            Bp=np.swapaxes(np.expand_dims(local_hand_point,0),1,2)
            RCp=np.swapaxes(np.matmul(camera2bottom_rot,Bp),1,2)
            # print(Bp.shape,RCp.shape,camera2bottom_trans.shape)
            Cp=RCp+np.expand_dims(camera2bottom_trans,1)

            grasp_center=np.concatenate((grasp_center,camera2grasp_trans),axis=0)
            grasp_pose=np.concatenate((grasp_pose,camera2grasp_rot),axis=0)
            grasp_score=np.concatenate((grasp_score,grasp_npy[:,[10]]),axis=0)
            hand_point=np.concatenate((hand_point,Cp),axis=0)

            local_hand_point_extend=GS.get_hand_points_extend(np.array([0,0,0]),np.array([1,0,0]),np.array([0,1,0]))
            bottom_center=grasp_center-GS.gripper.hand_depth*grasp_pose[:,:,0]
            break
    return grasp_center,bottom_center,grasp_pose,grasp_score,local_hand_point,local_hand_point_extend,hand_point

def collision_check_table_cuda(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,world2camera_rot,world2camera_trans):
    table_height=-1.5
    safe_dis=0.2
    before=len(grasp_center)
    bottom_center=torch.from_numpy(bottom_center).cuda()
    Gp=local_hand_point
    Gp=torch.transpose(torch.from_numpy(Gp).repeat(grasp_center.shape[0],1,1),1,2).cuda()

    pose_cuda=torch.from_numpy(grasp_pose).cuda()
    Cp=torch.matmul(pose_cuda,Gp)+bottom_center.reshape(-1,3,1)

    world2camera_rot_cuda=torch.from_numpy(world2camera_rot).repeat(grasp_center.shape[0],1,1).cuda()
    world2camera_trans_cuda=torch.from_numpy(world2camera_trans).cuda()

    Wp=torch.matmul(world2camera_rot_cuda,Cp)+world2camera_trans_cuda.reshape(1,3,1)

    lowest_points=torch.min(Wp[:,2,1:],dim=1,keepdim=True)[0]
    mask=lowest_points.cpu().numpy()>(table_height+safe_dis)
    mask=mask.flatten()

    bad_grasp_centers = grasp_center[~mask]  # (-1,3)
    bad_grasp_poses = grasp_pose[~mask]  # (-1,3,3)
    bad_bottom_centers = bottom_center[~mask]
    bad_hand_points = hand_point[~mask]

    grasp_center = grasp_center[mask] #(-1,3)
    grasp_pose = grasp_pose[mask]  #(-1,3,3)
    grasp_score = grasp_score[mask]#(-1,)
    bottom_center=bottom_center[mask]
    hand_point = hand_point[mask]

    after=len(grasp_center)
    print('Collision_check_table done:  ', before, ' to ', after)

def restrict_approach_angle(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,world2camera_rot,world2camera_trans):

        if grasp_center.shape[0] == 0:
            return 0
        max_angle=90
        before = len(grasp_center)

        # 抽取出各个抓取approach轴在世界坐标系W下的单位向量
        grasps_approach = grasp_pose[:, :, 0]  # (-1,3)
        # 单位化
        grasps_approach = grasps_approach / np.linalg.norm(grasps_approach, axis=1, keepdims=True)  # (-1,3)

        cos_angles = grasps_approach.dot(np.array([0, 0, -1]).T)  # (-1,)
        # 限制抓取approach轴与世界-z轴的角度范围
        mask = cos_angles > np.cos(max_angle / 180 * np.pi)
        # print(cos_angles[mask],np.cos(45/180*pi))
        bad_grasp_centers = grasp_center[~mask]
        bad_grasp_poses = grasp_pose[~mask]
        bad_hand_points = hand_point[~mask]
        bad_bottom_centers = bottom_center[~mask]

        grasp_centers = grasp_center[mask]
        grasp_poses = grasp_pose[mask]
        grasp_scores = grasp_score[mask]
        bottom_centers = bottom_center[mask]
        hand_points = hand_point[mask]

        after = len(grasp_centers)
        print('Restrict_approach_angl done:  ', before, ' to ', after)

def collision_check_pc_cuda(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,used_pc):
    minimum_points_num=10
    minimum_insert_dist=0.01
    if grasp_center.shape[0] == 0:
        return 0

    before = len(grasp_center)
    poses_cuda = torch.from_numpy(grasp_pose).cuda()
    mask_cuda = torch.zeros(grasp_center.shape[0]).cuda()
    local_hand_points = torch.from_numpy(local_hand_point).cuda()
    bottom_centers = torch.from_numpy(bottom_center).cuda()
    pc = torch.from_numpy(used_pc).cuda()

    gripper_points_p = torch.tensor([local_hand_points[4][0], local_hand_points[2][1], local_hand_points[1][2],
                                     local_hand_points[12][0], local_hand_points[9][1], local_hand_points[10][2],
                                     local_hand_points[3][0], local_hand_points[13][1], local_hand_points[2][2],
                                     local_hand_points[12][0], local_hand_points[15][1],
                                     local_hand_points[11][2]]).reshape(4, 1, -1).cuda()

    gripper_points_n = torch.tensor([local_hand_points[8][0], local_hand_points[1][1], local_hand_points[4][2],
                                     local_hand_points[10][0], local_hand_points[1][1], local_hand_points[9][2],
                                     local_hand_points[7][0], local_hand_points[2][1], local_hand_points[3][2],
                                     local_hand_points[20][0], local_hand_points[11][1],
                                     local_hand_points[12][2]]).reshape(4, 1, -1).cuda()

    # 对每个抓取进行碰撞检测
    for i in range(len(bottom_centers)):

        # 得到标准的旋转矩阵
        matrix = poses_cuda[i]
        # 转置=求逆（酉矩阵）
        grasp_matrix = matrix.T  # same as cal the inverse
        # 获取所有的点相对于夹爪底部中心点的向量
        points = pc - bottom_centers[i].reshape(1, 3)
        points_g = torch.mm(grasp_matrix, points.T).T
        # 查找左侧夹爪碰撞检查
        points_p = points_g.repeat(4, 1, 1)
        points_n = points_g.repeat(4, 1, 1)

        points_p = points_p - gripper_points_p
        points_n = points_n - gripper_points_n
        check_op = torch.where(torch.sum((torch.mul(points_p, points_n) < 0)[0], dim=1) == 3)[0]

        # check_c = (torch.mul(points_p,points_n)<0)[1:]
        check_ = torch.where(torch.sum((torch.mul(points_p, points_n) < 0)[1:], dim=2) == 3)[0]

        points_in_close_area = points_g[check_op]  # (-1,3)
        # if points_in_gripper_index.shape[0] == 0:#不存在夹爪点云碰撞
        if len(check_) == 0:
            collision = False
            # 检查夹爪内部点数是否够
            if len(points_in_close_area):  # 夹爪内部有点
                deepist_point_x = torch.min(points_in_close_area[:, 0])
                insert_dist = GS.hand_depth - deepist_point_x.cpu()
                # 设置夹爪内部点的最少点数,以及插入夹爪的最小深度
                if len(points_in_close_area) < minimum_points_num or insert_dist < minimum_insert_dist:
                    mask_cuda[i] = 1
            else:  # 夹爪内部根本没有点
                mask_cuda[i] = 1

        else:
            collision = True
            mask_cuda[i] = 1

    mask = mask_cuda.cpu()
    bad_grasp_centers = grasp_center[mask == 1]
    bad_grasp_poses = grasp_center[mask == 1]
    bad_hand_points = hand_point[mask == 1]
    bad_bottom_centers = bottom_centers[mask == 1]

    grasp_centers = grasp_center[mask == 0]
    grasp_poses = grasp_pose[mask == 0]
    grasp_scores = grasp_score[mask == 0]
    bottom_centers = bottom_centers[mask == 0]
    hand_points = hand_point[mask == 0]

    after = len(grasp_centers)
    print('Collision_check_pc done:  ', before, ' to ', after)

def show_pc_scene(mesh_list,hand_point,part_pos,part_rot_im,part_rot_real,camera2world_T):
    for idx,mesh in enumerate(mesh_list):
        if mesh=='001_1':
            mesh_file='/home/pika/assemble_scale_grasp_001/'+mesh+'/'+mesh+'.obj'
            mesh=o3d.io.read_triangle_mesh(mesh_file)

            raw_pc=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,np.asarray(mesh.vertices).shape[0]*5)
            voxel_pc=o3d.geometry.PointCloud.voxel_down_sample(raw_pc,0.001)
            pc=np.asarray(voxel_pc.points).T

            mesh_rot = np.empty([4])
            mesh_rot[0] = part_rot_real[idx]
            mesh_rot[1:] = part_rot_im[idx]
            world2mesh_T = RigidTransform(mesh_rot, part_pos[idx]).matrix

            print(idx,part_pos[idx])

            camera2mesh=np.matmul(camera2world_T,world2mesh_T)
            camera2mesh_rot=camera2mesh[0:3,0:3]
            camera2mesh_trans=camera2mesh[0:3,3].reshape(-1,3)

            pc=camera2mesh_rot.dot(pc).T+camera2mesh_trans

            print('camera2world trans:',camera2world_T[0:3,3].reshape(-1,3))
            print('world2mesh trans:',world2mesh_T[0:3,3].reshape(-1,3))
            print('camera2mesh trans:',camera2mesh_trans)
            print("x:", "min:",np.min(pc[:, 0]), "max:",np.max(pc[:, 0]),"mean:",np.mean(pc[:,0]))
            print("y:", "min:",np.min(pc[:,1]),  "max:",np.max(pc[:,1]), "mean:",np.mean(pc[:,1]))
            print("z:", "min:",np.min(pc[:,2]),  "max:",np.max(pc[:,2]), "mean:",np.mean(pc[:,2]))

            show_pc_mlab(pc,type='foreground')

            hand_point=hand_point[0]
            GS.show_grasp_3d(hand_point)
            show_pc_mlab(hand_point,type='gripper')
            mlab.axes()
            mlab.show()
            break

def show_pc_mlab(points,type='background'):
    if type=='foreground':
        color=(0,0,1)
    elif type=='gripper':
        color=(1,0,0)
    else:
        color=(0.5,0.5,0.5)

    print(points.shape)
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color, scale_factor=0.004)
    # mlab.points3d(points[:,0],points[:,1],points[:,2],color,scale_factor=0.004)

if __name__=='__main__':
    grasp_path='/home/pika/good_grasp/'
    obj_path='/home/pika/assemble_scale_grasp_001/'
    scene_path='/media/pika/Joyoyo/temp/1/'

    mesh='001_1'
    grasp_npy_name='default_'+mesh+'_140.npy'
    grasp_npy=np.load(os.path.join(grasp_path,grasp_npy_name))
    grasp_npy=grasp_npy[grasp_npy[:,-2]<=0.4]

    camera_pos=np.load(scene_path+'Camera_Pos.npy')
    camera_pos=camera_pos/100
    camera_rot_im=np.load(scene_path+'Camera_Rot_im.npy')
    camera_rot_real=np.load(scene_path+'Camera_Rot_rel.npy')

    part_pos=np.load(scene_path+'Parts_Pos.npy')
    part_pos=part_pos/100
    # part_pos[:,2]=-part_pos[:,2]
    part_rot_im=np.load(scene_path+'Parts_Rot_im.npy')
    part_rot_real=np.load(scene_path+'Parts_Rot_rel.npy')

    part_list=np.loadtxt(scene_path+'Total_Parts.txt',dtype='str')
    print(part_list)
    # grasp_center,bottom_center,grasp_pose,grasp_score,local_hand_point,local_hand_point_extend,hand_point=get_scene_grasp(part_list,grasp_npy,camera_pos,camera_rot_im,camera_rot_real,part_pos,part_rot_im,part_rot_real)

    random_index = 0
    camera_index = 0
    hand_depth = 0.125
    rgbd_idx = str(random_index).rjust(4, '0')

    # world to camera
    camera_rot = np.empty([4])
    camera_rot[0] = camera_rot_real[random_index][camera_index]
    camera_rot[1:] = camera_rot_im[random_index][camera_index]
    camera2world_trans = camera_pos[random_index][camera_index]
    camera2world_rot = RigidTransform.rotation_from_quaternion(camera_rot)

    camera2world_T = RigidTransform(camera2world_rot,camera2world_trans)
    camera2world = camera2world_T.matrix
    world2camera = camera2world_T.inverse().matrix
    print('camera2world:', camera2world)
    print('world2camera trans:', world2camera)


    pcd=o3d.io.read_point_cloud(scene_path+'0001.ply')
    temp=np.asarray(pcd.points)
    # temp[:,2]=-temp[:,2]
    pcd.points=o3d.utility.Vector3dVector(temp)
    # pcd_hand_points=o3d.geometry.PointCloud()
    # hand_point[:,:,2]=-hand_point[:,:,2]
    # pcd_hand_points.points=o3d.utility.Vector3dVector(hand_point[0])
    # pcd_hand_points.paint_uniform_color([1,0,0])

    #

    idx=np.where(part_list==mesh)[0][0]
    print('idx:',idx)
    mesh_file = '/home/pika/assemble_scale_grasp_001/' + mesh + '/' + mesh + '.obj'
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 5)
    voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
    pc = np.asarray(voxel_pc.points).T

    raw_pcd=o3d.geometry.PointCloud()
    raw_pcd.points=o3d.utility.Vector3dVector(pc.T)
    axis_pcd_0=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

    for i in range(10):
        mesh_rot = np.empty([4])
        mesh_rot[0] = part_rot_real[i]
        mesh_rot[1:] = part_rot_im[i]
        # print("part_rot:", part_list[i],mesh_rot, quaternion2euler(mesh_rot[[1, 2, 3, 0]]))
        # print("part_pos", part_pos[i])

    mesh_rot = np.empty([4])
    mesh_rot[0] = part_rot_real[idx]
    mesh_rot[1:] = part_rot_im[idx]
    print("part_rot:", part_pos[idx],mesh_rot,quaternion2euler(mesh_rot[[1,2,3,0]]))
    mesh_rot = RigidTransform.rotation_from_quaternion(mesh_rot)

    mesh2world = RigidTransform(mesh_rot, part_pos[idx]).matrix
    print('mesh2world:',mesh2world)
    # rotate_matrix=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    # mesh_90=RigidTransform(rotate_matrix,np.zeros(3)).matrix
    #
    # world2mesh_T[0:3,0:3]=np.dot(world2mesh_T[0:3,0:3],rotate_matrix)


    camera2mesh = np.matmul(world2camera, mesh2world)

    camera2mesh_rot = camera2mesh[0:3, 0:3]
    camera2mesh_trans = camera2mesh[0:3, 3].reshape(-1, 3)
    print('camera2mesh:',camera2mesh)

    pc = np.dot(camera2mesh_rot,pc).T + camera2mesh_trans
    # pc = np.dot(euler2rot([90,0,72]),pc).T #+ camera2mesh_trans
    # pc = np.dot(pc.T,euler2rot([-90,0,72]).T) #+ camera2mesh_trans

    pc[:,2]=-pc[:,2]

    center=np.dot(world2camera[0:3,0:3],part_pos[idx].T).T+world2camera[0:3,3]
    print('center:',center)
    center[2]=-center[2]
    print("x:", "min:", np.min(pc[:, 0]), "max:", np.max(pc[:, 0]), "mean:", np.mean(pc[:, 0]))
    print("y:", "min:", np.min(pc[:, 1]), "max:", np.max(pc[:, 1]), "mean:", np.mean(pc[:, 1]))
    print("z:", "min:", np.min(pc[:, 2]), "max:", np.max(pc[:, 2]), "mean:", np.mean(pc[:, 2]))

    print()
    # temp=pc[:,0].copy()
    # pc[:,0]=pc[:,2]
    # pc[:,2]=temp
    #
    # temp=center[0].copy()
    # center[0]=center[2]
    # center[2]=temp

    # print("center:",center)
    # print("x:", "min:", np.min(pc[:, 0]), "max:", np.max(pc[:, 0]), "mean:", np.mean(pc[:, 0]))
    # print("y:", "min:", np.min(pc[:, 1]), "max:", np.max(pc[:, 1]), "mean:", np.mean(pc[:, 1]))
    # print("z:", "min:", np.min(pc[:, 2]), "max:", np.max(pc[:, 2]), "mean:", np.mean(pc[:, 2]))

    mesh_pcd=o3d.geometry.PointCloud()
    mesh_pcd.points=o3d.utility.Vector3dVector(pc)
    # mesh_pcd=copy.deepcopy(raw_pcd)
    # mesh_pcd.rotate(camera2mesh[0:3,0:3])
    # mesh_pcd.translate([camera2mesh[0,3],camera2mesh[1,3],-camera2mesh[2,3]])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=center)


    # raw_scene=np.load('/media/pika/Joyoyo/Omniverse_NewNew/1/pointcloud_0001.npy').T
    # raw_scene=raw_scene/100
    # print(raw_scene.shape)


    # rotate_matrix=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    # rotate_matrix=np.array([[0,0,-1],[0,1,0],[1,0,0]])
    # rotate_matrix=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    # rotate_matrix=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    # mesh_90=RigidTransform(rotate_matrix,np.zeros(3)).matrix

    # rotate_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # mesh_90_2=RigidTransform(rotate_matrix,np.zeros(3)).matrix

    # mesh_90=np.matmul(mesh_90,mesh_90_2)

    # rotate_matrix=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # mesh_90 = RigidTransform(rotate_matrix, np.zeros(3)).matrix
    # camera2world_T=np.matmul(camera2world_T,mesh_90)

    # camera2world_T=np.matmul(camera2world_T,world_90_2)
    # camera2world_trans=camera2world_T[0:3,3]
    # camera2world_rot=camera2world_T[0:3,0:3]
    # raw_scene=camera2world_rot.dot(raw_scene).T+camera2world_trans
    # raw_scene_pc=o3d.geometry.PointCloud()
    # raw_scene_pc.points=o3d.utility.Vector3dVector(raw_scene)
    #
    # img_path = '/media/pika/Joyoyo/Omniverse_NewNew/1/'
    # raw_scene_color=Image.open(img_path+'rgb_0001.png')
    # img_array = np.array(raw_scene_color)
    # img_array = img_array[:, :, 0:3].reshape(-1, 3) / 255
    # raw_scene_pc.colors=o3d.utility.Vector3dVector(img_array)

    # x:red, y:green. z:blue
    o3d.visualization.draw_geometries([pcd]+[]+[axis_pcd]+[]+[mesh_pcd])
    # o3d.visualization.draw_geometries([raw_pcd]+[]+[axis_pcd_0]+[mesh_pcd]+[])

    # point_cloud_original=np.load('/media/pika/Joyoyo/Omniverse_NewNew/1/RenderProduct_Replicator/')

    # print(bottom_center[0],part_pos[idx])
    #
    # collision_check_table_cuda(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,world2camera_rot,world2camera_trans)
    #
    # restrict_approach_angle(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,world2camera_rot,world2camera_trans)
    #
    # # show_pc_scene(part_list,hand_point,part_pos,part_rot_im,part_rot_real,camera2world_T)
    #
    # pc=o3d.io.read_point_cloud('/home/pika/Desktop/0001.ply')
    # used_pc=np.array(pc.points)
    # used_pc[:,2]=used_pc[:,2]
    # # used_pc=-used_pc
    # collision_check_pc_cuda(grasp_center,bottom_center,local_hand_point,hand_point,grasp_pose,grasp_score,used_pc)
    #
    # # show_grasp_pc()
