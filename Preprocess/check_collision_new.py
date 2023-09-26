import numpy as np
import os,sys

if __name__=='__main__':
    data_path='/media/pika/Joyoyo/0921/'
    source_path='/home/pika/assemble_step/'

    for part in os.listdir(source_path):
        for i in range(1,11):
            dataset_path=data_path+part+'_'+str(i)
            # print(dataset_path)
            part_list=np.loadtxt(dataset_path+'/'+'Total_Parts.txt',dtype=str)
            for part_name in part_list:
                part_npy=part_name+'_collision_new.npy'
                datum=np.load(dataset_path+'/'+part_npy)
                if len(datum)==0:
                    print("empty collision of ",part_name,' in ',dataset_path)
                if part_npy not in os.listdir(dataset_path):
                    print("error happened in ",dataset_path,'lack of ',part_name)