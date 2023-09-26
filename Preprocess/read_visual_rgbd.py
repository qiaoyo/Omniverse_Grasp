import numpy as np
import os,sys
from PIL import Image
import open3d as o3d
import cv2
import yaml

def depth2pc(depth,camera):
    depth=np.ravel(depth)
    pc=np.zeros((depth.shape[0],3))
    for i in range(depth.shape[0]):
        z=depth[i]
        x=depth[i]*(i%1280-camera['cx'])/camera['fx']
        y=depth[i]*(i//1280-camera['cy'])/camera['fy']
        pc[i]=np.array([x,y,z])
    return pc

if __name__=='__main__':

    PART=os.listdir('/home/pika/assemble_step/')
    for part in PART:
        print(part,':')
        for j in range(1,11):
            data_path = '/media/pika/Joyoyo/0921/'+part+'_'+str(j)+'/'
            grasp_path = '/home/pika/good_grasp/'
            camera_name_list = ['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0', 'RenderProduct_Replicator',
                                'RenderProduct_Replicator_01', 'RenderProduct_Replicator_02', 'RenderProduct_Replicator_03',
                                'RenderProduct_Replicator_04', 'RenderProduct_Replicator_05', 'RenderProduct_Replicator_06']
            with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.0/Omniverse_Grasp/config.yaml') as Config_file:
                Config_yaml = yaml.load(Config_file, Loader=yaml.FullLoader)
            simulation_steps = Config_yaml['Renderer']['Simulation_steps']
            render_steps = Config_yaml['Renderer']['Render_steps']
            camera_num = Config_yaml['Camera']['num']
            mesh_path = '/home/pika/assemble_scale_grasp_001/'
            part_list = np.loadtxt(data_path + 'Total_Parts.txt', dtype=str)

            show_mesh_pcd = False

            camera_pos = np.load(data_path + 'Camera_Pos.npy')
            camera_pos = camera_pos / 100
            camera_rot_real = np.load(data_path + 'Camera_Rot_rel.npy')
            camera_rot_im = np.load(data_path + 'Camera_Rot_im.npy')

            part_pos = np.load(data_path + 'Parts_Pos.npy')
            part_pos = part_pos / 100
            part_rot_real = np.load(data_path + 'Parts_Rot_rel.npy')
            part_rot_im = np.load(data_path + 'Parts_Rot_im.npy')

            for camera_idx in range(camera_num):
                for random_idx in range(render_steps):
                    camera_name_list=['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0','RenderProduct_Replicator','RenderProduct_Replicator_01','RenderProduct_Replicator_02','RenderProduct_Replicator_03','RenderProduct_Replicator_04','RenderProduct_Replicator_05','RenderProduct_Replicator_06']
                    # camera_name_list=['RenderProduct_Replicator','RenderProduct_omni_kit_widget_viewport_ViewportTexture_0','RenderProduct_Replicator_01','RenderProduct_Replicator_02','RenderProduct_Replicator_03','RenderProduct_Replicator_04','RenderProduct_Replicator_05','RenderProduct_Replicator_06']
                    # data_path='/media/pika/Joyoyo/temp/1/'
                    img_path=data_path+camera_name_list[camera_idx]+'/rgb/'
                    depth_path=data_path+camera_name_list[camera_idx]+'/distance_to_camera/'

                    camera={
                        'cx' : 640,
                        'cy' : 360,
                        'fx' : 660*10,#6430,
                        'fy' : 660*10,#6430,
                    }
                    # camera_pos shape: [rep_frame, camera_num, 3], camera_rot_im shape: [rep_frame, camera_num, 3], camera_rot_real shape: [rep_frame, camera_num]
                    # part_pos shape: [part_idx, 3], part_rot_im shape: [part_idx, 3], part_rot_real shape: [part_idx, 3]


                    rgbd_idx=str(random_idx+1).rjust(4,'0')

                    img=Image.open(img_path+'rgb_'+rgbd_idx+'.png')
                    depth=np.load(depth_path+'distance_to_camera_'+rgbd_idx+'.npy')
                    camera_pos=np.load(os.path.join(data_path,'Camera_Pos.npy'))
                    img_array=np.array(img)
                    #
                    # print(np.min(depth),np.max(depth))
                    # x:red, y:green. z:blue
                    # axis_pcd2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
                    img=img_array[:,:,0:3].reshape(-1,3)/255
                    # print(np.min(img),np.max(img),img.shape)

                    pc=depth2pc(depth,camera)
                    # print("x:",np.min(pc[:,0]),np.max(pc[:,0]))
                    # print("y:",np.min(pc[:,1]),np.max(pc[:,1]))
                    # print("z:",np.min(pc[:,2]),np.max(pc[:,2]))
                    # print(pc.shape)
                    #
                    pcd=o3d.geometry.PointCloud()
                    pcd.points=o3d.utility.Vector3dVector(pc)
                    pcd.colors=o3d.utility.Vector3dVector(img)
                    o3d.io.write_point_cloud(depth_path+rgbd_idx+'.ply',pcd)
                    print(depth_path+rgbd_idx+'.ply'+' done')

    #     # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,4])
    #     #
    #     o3d.visualization.draw_geometries([pcd]+[])
    # data_path = '/media/pika/Joyoyo/ttemp/1/'
    # img_path = data_path
    # depth_path = data_path
    #
    # camera = {
    #     'cx': 640,
    #     'cy': 360,
    #     'fx': 660 * 10,  # 6430,
    #     'fy': 660 * 10,  # 6430,
    # }
    # # camera_pos shape: [rep_frame, camera_num, 3], camera_rot_im shape: [rep_frame, camera_num, 3], camera_rot_real shape: [rep_frame, camera_num]
    # # part_pos shape: [part_idx, 3], part_rot_im shape: [part_idx, 3], part_rot_real shape: [part_idx, 3]
    # random_idx=0
    # rgbd_idx = str(random_idx + 1).rjust(4, '0')
    #
    # img = Image.open(img_path + 'rgb_' + rgbd_idx + '.png')
    # depth = np.load(depth_path + 'distance_to_camera_' + rgbd_idx + '.npy')
    # camera_pos = np.load(os.path.join(data_path, 'Camera_Pos.npy'))
    # img_array = np.array(img)
    # #
    # print(np.min(depth), np.max(depth))
    # # x:red, y:green. z:blue
    # # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-0.097868 , -0.014888,  0.037968])
    # # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-0.297868 , -0.014888,  0])
    # # axis_pcd2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    # img = img_array[:, :, 0:3].reshape(-1, 3) / 255
    # print(np.min(img), np.max(img), img.shape)
    #
    # pc = depth2pc(depth, camera)
    # print("x:", np.min(pc[:, 0]), np.max(pc[:, 0]))
    # print("y:", np.min(pc[:, 1]), np.max(pc[:, 1]))
    # print("z:", np.min(pc[:, 2]), np.max(pc[:, 2]))
    # print(pc.shape)
    # #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(img)
    # o3d.io.write_point_cloud(depth_path + rgbd_idx + '.ply', pcd)
    # pcd=o3d.io.read_point_cloud(img_path[:-4]+'0001.ply')
    # xyz=np.array(pcd.points)
    # img=img_array[:,:,0:3].reshape(-1,3)/255
    # print(np.min(img),np.max(img),img.shape)
    # pcd.colors=o3d.utility.Vector3dVector(img)

    # pcd=pcd.random_down_sample(sampling_ratio=0.2)
    # idx=pcd[1]
    # pcd=pcd[0]
    # print(idx.shape)

    #
    # xyz=np.load('/media/pika/Joyoyo/Omniverse_NewNew/1/pointcloud_0001.npy')
    # print(xyz.shape)
    # xyz=xyz/100
    # print("x:",np.min(xyz[:,0]),np.max(xyz[:,0]))
    # print("y:",np.min(xyz[:,1]),np.max(xyz[:,1]))
    # print("z:",np.min(xyz[:,2]),np.max(xyz[:,2]))
    # pcd2=o3d.geometry.PointCloud()
    # pcd2.points=o3d.utility.Vector3dVector(xyz)
#
#     # # pcd.paint_uniform_color([1, 0.706, 0])
# #     # # pcd2.paint_uniform_color([0, 0.651, 0.929])
#     axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
#     pc[:,2]=pc[:,2]-4
#     pcd.points=o3d.utility.Vector3dVector(pc)
#     o3d.visualization.draw_geometries([pcd2,axis_pcd,axis_pcd2])

# if __name__=='__main__':
#     img_path='/media/pika/新加卷/dreds_test/shapenet_generate_1216_val_novel/00000/'
#     img=cv2.imread(img_path+'0000_color.png')
#
#     img=cv2.resize(img,(640,360))
#
#     os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
#     depth=cv2.imread(img_path+'0000_depth_120.exr',cv2.IMREAD_UNCHANGED)
#     print(img.shape,depth.shape)
#     # assert (depth[:,0]==depth[:,1]).all()
#     # assert (depth[:,1]==depth[:,2]).all()
#     camera = {
#                 'cx' : 180-0.5,
#                 'cy' : 320-0.5,
#                 'fx' : 6430,#64300,
#                 'fy' : 6430,#64300,
#             }
#     pc=depth2pc(depth,camera=camera)
#     print("x:",np.min(pc[:,0]),np.max(pc[:,0]))
#     print("y:",np.min(pc[:,1]),np.max(pc[:,1]))
#     print("z:",np.min(pc[:,2]),np.max(pc[:,2]))
#
#     pcd=o3d.geometry.PointCloud()
#     pcd.points=o3d.utility.Vector3dVector(pc)
#
#     o3d.visualization.draw_geometries([pcd])
#     #
#     #