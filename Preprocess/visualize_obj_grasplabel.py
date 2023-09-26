import copy

from graspnetAPI import GraspNet
import os,sys
import numpy as np
import open3d as o3d
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup, RECT_GRASP_ARRAY_LEN
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix


class Grasp():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id

        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        - the length of the numpy array is 17.
        '''
        if len(args) == 0:
            self.grasp_array = np.array([0, 0.02, 0.02, 0.02, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=np.float64)
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == 7:
            score, width, height, depth, rotation_matrix, translation, object_id = args
            self.grasp_array = np.concatenate(
                [np.array((score, width, height, depth)), rotation_matrix.reshape(-1), translation,
                 np.array((object_id)).reshape(-1)]).astype(np.float64)
        else:
            raise ValueError('only 1 or 7 arguments are accepted')

    def __repr__(self):
        return 'Grasp: score:{}, width:{}, height:{}, depth:{}, translation:{}\nrotation:\n{}\nobject id:{}'.format(
            self.score, self.width, self.height, self.depth, self.translation, self.rotation_matrix, self.object_id)

    @property
    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return float(self.grasp_array[0])

    @score.setter
    def score(self, score):
        '''
        **input:**

        - float of the score.
        '''
        self.grasp_array[0] = score

    @property
    def width(self):
        '''
        **Output:**

        - float of the width.
        '''
        return float(self.grasp_array[1])

    @width.setter
    def width(self, width):
        '''
        **input:**

        - float of the width.
        '''
        self.grasp_array[1] = width

    @property
    def height(self):
        '''
        **Output:**

        - float of the height.
        '''
        return float(self.grasp_array[2])

    @height.setter
    def height(self, height):
        '''
        **input:**

        - float of the height.
        '''
        self.grasp_array[2] = height

    @property
    def depth(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return float(self.grasp_array[3])

    @depth.setter
    def depth(self, depth):
        '''
        **input:**

        - float of the depth.
        '''
        self.grasp_array[3] = depth

    @property
    def rotation_matrix(self):
        '''
        **Output:**

        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.grasp_array[4:13].reshape((3, 3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of matrix

        - len(args) == 9: float of matrix
        '''
        if len(args) == 1:
            self.grasp_array[4:13] = np.array(args[0], dtype=np.float64).reshape(9)
        elif len(args) == 9:
            self.grasp_array[4:13] = np.array(args, dtype=np.float64)

    @property
    def translation(self):
        '''
        **Output:**

        - np.array of shape (3,) of the translation.
        '''
        return self.grasp_array[13:16]

    @translation.setter
    def translation(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y, z

        - len(args) == 3: float of x, y, z
        '''
        if len(args) == 1:
            self.grasp_array[13:16] = np.array(args[0], dtype=np.float64)
        elif len(args) == 3:
            self.grasp_array[13:16] = np.array(args, dtype=np.float64)

    @property
    def object_id(self):
        '''
        **Output:**

        - int of the object id that this grasp grasps
        '''
        return int(self.grasp_array[16])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **Input:**

        - int of the object_id.
        '''
        self.grasp_array[16] = object_id

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)

        **Output:**

        - Grasp instance after transformation, the original Grasp will also be changed.
        '''
        rotation = T[:3, :3]
        translation = T[:3, 3]
        self.translation = np.dot(rotation, self.translation.reshape((3, 1))).reshape(-1) + translation
        self.rotation_matrix = np.dot(rotation, self.rotation_matrix)
        return self

    def to_open3d_geometry(self, color=None):
        '''
        **Input:**

        - color: optional, tuple of shape (3) denotes (r, g, b), e.g., (1,0,0) for red

        **Ouput:**

        - list of open3d.geometry.Geometry of the gripper.
        '''
        return plot_gripper_pro_max(self.translation, self.rotation_matrix, self.width, self.depth, score=self.score,
                                    color=color)

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

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    '''
    Author: chenxi-wang

    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

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

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def grasp2geometry(grasp):
    grasp=Grasp(grasp)
    geometry=grasp.to_open3d_geometry()
    return [geometry]

def visualizeGrasp(mesh_name,num_sample=20,random=False):
    meshs_path = '/home/pika/object-grasp-annotation/models/'
    # mesh_name='001_1'
    mesh_pcd=o3d.io.read_point_cloud(meshs_path+mesh_name+'/'+'nontextured.ply')
    axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

    grasp_label = np.load('/home/pika/object-grasp-annotation/grasp_label/' + mesh_name+'_labels.npz')

    graspLabels={}
    graspLabels[0] = (grasp_label['points'].astype(np.float32), grasp_label['offsets'].astype(np.float32), grasp_label['scores'].astype(np.float32))
    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    grasp_group = []
    obj_idx=0
    sampled_points, offsets, fric_coefs = graspLabels[obj_idx]
    point_inds = np.arange(sampled_points.shape[0])

    num_points = len(point_inds)
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    fric_coef_thresh=0.2
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
    grasp_group.append(obj_grasp_array)
    grasp_group=np.array(grasp_group).squeeze()
    if random:
        np.random.shuffle(grasp_group)
    grasp_group=copy.deepcopy(grasp_group[0:num_sample])
    grippers=[]
    for sample_garsp in grasp_group:
        grippers+=grasp2geometry(sample_garsp)

    # grippers+=grasp_group.random_sample(numGrasp=20).to_open3d_geometry_list()
    return grasp_group,grippers


if __name__=='__main__':
    # mesh_name='001_1'
    # num_sample=20
    # grippers=visualizeGrasp(mesh_name,num_sample)
    # print(np.array(grippers[0].vertices))
    # meshs_path = '/home/pika/object-grasp-annotation/models/'
    # # mesh_name='001_1'
    # mesh_pcd=o3d.io.read_point_cloud(meshs_path+mesh_name+'/'+'nontextured.ply')
    axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    # o3d.visualization.draw_geometries(grippers+[mesh_pcd]+[axis_pcd])
    # x:red, y:green. z:blue
    grasp=np.zeros(17)
    grasp=Grasp()
    g=grasp.to_open3d_geometry()
    vertices=np.array(g.vertices)
    vertices_pcd=o3d.geometry.PointCloud()
    tt=28
    vertices_pcd.points=o3d.utility.Vector3dVector([vertices[tt],vertices[tt+1],vertices[tt+2],vertices[tt+3]])
    temp_color = np.zeros((4, 3))
    temp_color[0]=np.array([1,0,0])
    temp_color[1]=np.array([0,1,0])
    temp_color[2]=np.array([0,0,1])
    temp_color[3]=np.array([1,1,1])
    vertices_pcd.colors=o3d.utility.Vector3dVector(temp_color)
    o3d.visualization.draw_geometries([g]+[axis_pcd]+[vertices_pcd])

# if __name__=='__main__':
#     grasp_label_path='/home/pika/object-grasp-annotation/grasp_label/'
#     meshs_path='/home/pika/object-grasp-annotation/models/'
#
#     # cnt=0
#     # for mesh_name in os.listdir('/home/pika/grasp_label/'):
#     #     # mesh_name='001_1'
#     #     # mesh_path=meshs_path+mesh_name+'/'
#     #     grasp_label = np.load('/home/pika/grasp_label/' + mesh_name )
#     #     points=grasp_label['points']
#     #     cnt+=1
#     #     print(mesh_name,points.shape,cnt)
#
#     # offsets=grasp_label['offsets']
#     # collision=grasp_label['collision']
#     # scores=grasp_label['scores']
#     mesh_name='001_1'
#     mesh_pcd=o3d.io.read_point_cloud(meshs_path+mesh_name+'/'+'nontextured.ply')
#     axis_pcd=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
#
#     grasp_label = np.load('/home/pika/object-grasp-annotation/grasp_label/' + mesh_name+'_labels.npz')
#
#     grippers=[]
#
#     graspLabels={}
#     graspLabels[0] = (grasp_label['points'].astype(np.float32), grasp_label['offsets'].astype(np.float32), grasp_label['scores'].astype(np.float32))
#     num_views, num_angles, num_depths = 300, 12, 4
#     template_views = generate_views(num_views)
#     template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
#     template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])
#
#     grasp_group = GraspGroup()
#     obj_idx=0
#     sampled_points, offsets, fric_coefs = graspLabels[obj_idx]
#     point_inds = np.arange(sampled_points.shape[0])
#
#     num_points = len(point_inds)
#     target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
#     target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
#     views = np.tile(template_views, [num_points, 1, 1, 1, 1])
#     angles = offsets[:, :, :, :, 0]
#     depths = offsets[:, :, :, :, 1]
#     widths = offsets[:, :, :, :, 2]
#
#     fric_coef_thresh=0.2
#     mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0))
#     target_points = target_points[mask1]
#     views = views[mask1]
#     angles = angles[mask1]
#     depths = depths[mask1]
#     widths = widths[mask1]
#     fric_coefs = fric_coefs[mask1]
#     Rs = batch_viewpoint_params_to_matrix(-views, angles)
#
#     GRASP_HEIGHT = 0.02
#     num_grasp = widths.shape[0]
#     scores = (1.1 - fric_coefs).reshape(-1, 1)
#     widths = widths.reshape(-1, 1)
#     heights = GRASP_HEIGHT * np.ones((num_grasp, 1))
#     depths = depths.reshape(-1, 1)
#     rotations = Rs.reshape((-1, 9))
#     object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)
#
#     obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(
#         np.float32)
#     grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))
#
#     grippers+=grasp_group.random_sample(numGrasp=20).to_open3d_geometry_list()
#
#
#     o3d.visualization.draw_geometries(grippers+[mesh_pcd]+[axis_pcd])