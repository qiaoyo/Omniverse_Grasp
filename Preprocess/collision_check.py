import numpy as np
import open3d as o3d
import os,sys
import yaml
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup, RECT_GRASP_ARRAY_LEN
import copy
class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, collision,approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False,
               empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious

            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        collision_idx=np.where(collision==False)
        T = grasp_group.translations[collision_idx]
        R = grasp_group.rotation_matrices[collision_idx]
        heights = grasp_group.heights[:, np.newaxis][collision_idx]
        depths = grasp_group.depths[:, np.newaxis][collision_idx]
        widths = grasp_group.widths[:, np.newaxis][collision_idx]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths))
        mask3 = (targets[:, :, 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask5 = (targets[:, :, 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[:, :, 1] > widths / 2)
        # bottom mask
        mask7 = ((targets[:, :, 0] <= depths - self.finger_length) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:, :, 0] <= depths - self.finger_length - self.finger_width) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        bottom_volume = (
                    heights * (widths + 2 * self.finger_width) * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        shifting_volume = (heights * (widths + 2 * self.finger_width) * approach_dist / (self.voxel_size ** 3)).reshape(
            -1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # print(global_iou)
        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        collision_mask_full=np.ones(len(grasp_group.grasp_group_array),dtype=bool)
        collision_mask_full[collision_idx]=collision_mask

        if not (return_empty_grasp or return_ious):
            return collision_mask_full

        ret_value = [collision_mask, ]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size ** 3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1) / inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume + 1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume + 1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume + 1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume + 1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value

def graspnpy2graspgroup(grasp_label,collision=[]):
    '''

    :param grasp_label: the npz file
    :return: GraspGroup class. GraspGroup.grasp_group_array is the grasp label. shape: [n,17]
    '''
    graspLabels = {}
    graspLabels[0] = (grasp_label['points'].astype(np.float32), grasp_label['offsets'].astype(np.float32),
                      grasp_label['scores'].astype(np.float32))
    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    grasp_group_array = []
    obj_idx = 0
    sampled_points, offsets, fric_coefs = graspLabels[obj_idx]
    point_inds = np.arange(sampled_points.shape[0])

    num_points = len(point_inds)
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    fric_coef_thresh = 0.2
    mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0))
    target_points = target_points[mask1]
    views = views[mask1]
    angles = angles[mask1]
    depths = depths[mask1]
    widths = widths[mask1]
    fric_coefs = fric_coefs[mask1]
    Rs = batch_viewpoint_params_to_matrix(-views, angles)

    GRASP_HEIGHT = 0.02
    num_grasp = widths.shape[0]
    scores = (1.1 - fric_coefs).reshape(-1, 1)
    widths = widths.reshape(-1, 1)
    heights = GRASP_HEIGHT * np.ones((num_grasp, 1))
    depths = depths.reshape(-1, 1)
    rotations = Rs.reshape((-1, 9))
    object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)

    obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(
        np.float32)
    grasp_group_array.append(obj_grasp_array)
    grasp_group_array = np.array(grasp_group_array).squeeze()

    if len(collision) != 0:
        grasp_group_array=grasp_group_array[collision==False]

    graspGroup=GraspGroup()
    graspGroup.grasp_group_array=grasp_group_array

    return graspGroup

if __name__=='__main__':
    with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.0/Omniverse_Grasp/config.yaml') as Config_file:
        Config_yaml = yaml.load(Config_file, Loader=yaml.FullLoader)
    simulation_steps = Config_yaml['Renderer']['Simulation_steps']
    render_steps = Config_yaml['Renderer']['Render_steps']

    camera_idx = 0
    random_idx = 0  # 0~9
    rgbd_idx = str(random_idx + 1).rjust(4, '0')  # 0001~0010
    camera_name_list = ['RenderProduct_omni_kit_widget_viewport_ViewportTexture_0', 'RenderProduct_Replicator',
                        'RenderProduct_Replicator_01', 'RenderProduct_Replicator_02', 'RenderProduct_Replicator_03',
                        'RenderProduct_Replicator_04', 'RenderProduct_Replicator_05', 'RenderProduct_Replicator_06']

    data_path = '/media/pika/Joyoyo/temp/1/'
    grasp_path = '/home/pika/good_grasp/'
    img_path = data_path + camera_name_list[camera_idx] + '/rgb/'
    depth_path = data_path + camera_name_list[camera_idx] + '/distance_to_camera/'

    part_list = np.loadtxt(data_path + 'Total_Parts.txt', dtype=str)
    mesh_name = '001_2'

    scene_pcd = o3d.io.read_point_cloud(depth_path + rgbd_idx+'_'+mesh_name+'.ply')
    scene_point_cloud=np.array(scene_pcd.points)
    axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
    # o3d.visualization.draw_geometries([axis_pcd]+[scene_pcd])

    grasp_npy_path='/home/pika/object-grasp-annotation/grasp_label/'

    grasp_label=np.load(grasp_npy_path+mesh_name+'_labels.npz')
    grasp_Group=graspnpy2graspgroup(grasp_label)

    mfcDetector=ModelFreeCollisionDetector(scene_point_cloud)
    collision_mask=mfcDetector.detect(grasp_Group,approach_dist=0.03)
    print(collision_mask.shape,(collision_mask==True).sum(),(collision_mask==False).sum())

    grasp_label=copy.deepcopy(grasp_Group.grasp_group_array)
    collision_idx=np.where(collision_mask==True)
    collision_idx_n=np.where(collision_mask==False)
    grasp_label_with_collision=grasp_label[collision_idx]
    grasp_label_without_collision=grasp_label[collision_idx_n]
    print(grasp_label_with_collision.shape,grasp_label_without_collision.shape)

    from visualize import npy2geometry
    print(collision_idx[0])
    print(collision_idx_n[0])