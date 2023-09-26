import numpy as np
import os,sys

data_path='/home/pika/object-grasp-annotation/grasp_label/001_1_labels.npz'
data=np.load(data_path)
points=data['points']
offsets=data['offsets']
collision=data['collision']
print(np.sum(collision==False),np.sum(collision==True))

scores=data['scores']

