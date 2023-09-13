import os
import shutil

path='/home/pika/assemble_scale_grasp_001/'
target_path='/home/pika/assemble_scale_grasp/'

for file_name in os.listdir(path):
    ply_file=os.path.join(path,file_name,file_name+'.ply')
    target_ply_file=os.path.join(target_path,file_name,file_name+'.ply')
    shutil.copyfile(ply_file,target_ply_file)
