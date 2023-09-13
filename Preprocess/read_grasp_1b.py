import numpy as np
import os,sys

if __name__=='__main__':
    grasp_label_path='/media/pika/Elements/graspnet-1Billion/grasp_label'
    name='000_labels.npz'

    collision_label_path='/media/pika/Elements/graspnet-1Billion/collision_label/'
    scene_name='scene_0000'

    grasp_npy=np.load(os.path.join(grasp_label_path,name))
    print(grasp_npy.files)
    print(grasp_npy['points'].shape)
    print(grasp_npy['offsets'].shape)
    print(grasp_npy['collision'].shape)
    print(grasp_npy['scores'].shape)

    collision_npy=np.load(os.path.join(collision_label_path,scene_name+'/'+'collision_labels.npz'))
    print(collision_npy.files)
    print(collision_npy['arr_8'].shape)
    # print(grasp_npy.shape)