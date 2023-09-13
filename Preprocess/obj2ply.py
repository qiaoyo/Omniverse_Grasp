#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:47:09 2020

@author: oshi

"""
import os.path
import random

import numpy as np
import aspose.threed as a3d
# filename = "/home/oshi/facegen/Face/face_test_new.obj"

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        self.vt = []
        self.vn = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    # vertex = (round(vertex[0], 6), round(vertex[1], 6), round(vertex[2], 6))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            num = int(string[i:-1].split("/")[0])
                            face.append(num - 1)
                            break
                        num = int(string[i:string.find(" ", i)].split("/")[0])
                        face.append(num - 1)
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))
                elif line[:2] == "vt":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    vt_vertex = (float(line[index1:index2]), float(line[index2:index3]))
                    # vt_vertex = (round(vertex[0], 2), round(vertex[1], 2))
                    self.vt.append(vt_vertex)
                elif line[:2] == "vn":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex_vn = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    # vertex_vn = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vn.append(vertex_vn)

            f.close()
        except IOError:
            print(".obj file not found.")

    def get_Vertices(self):
        return self.vertices

    def set_vertices(self, vertices):
        self.vertices = vertices

    def write_ply(self, outName):
        vert = np.array(self.vertices)
        vn = np.array(self.vn)
        vt = np.array(self.vt)
        line_property = np.hstack((vert, vn, vt))

        faces = np.array(self.faces)
        faces = np.hstack((np.ones((faces.shape[0], 1)) * 3, faces))

        num_faces = len(faces)
        num_vertices = len(vert)

        header = ""
        header += "ply\nformat ascii 1.0\n"
        header += "comment Created by Arpit Maclay@Vidalign\n"
        header += "element vertex " + str(num_vertices) + "\n"
        header += "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n"  # "property uchar red\nproperty uchar greenp\nroperty uchar blue\nproperty uchar alpha\n"
        header += "element face " + str(num_faces) + "\n"
        header += "property list uchar uint vertex_indices\nend_header\n"

        with open(str.encode(outName), 'wb') as f:
            f.write(str.encode(header))
            np.savetxt(f, line_property, delimiter=" ", fmt='%f')
            np.savetxt(f, faces, delimiter=" ", fmt='%i')

# the a3d.scenes can be exceeded. so randomize the file_list and run several times

def main():
    file_path='/home/pika/assemble_scale_grasp_001/'
    n=0
    file_list=os.listdir(file_path)
    random.shuffle(file_list)
    for file_name in file_list:
        print(file_name,n)
        #file_name='001_1'
        scene=a3d.Scene.from_file(os.path.join(file_path,file_name,file_name+'.obj'))
        scene.save(os.path.join(file_path,file_name,file_name+'.ply'))
        n+=1

if __name__=='__main__':
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
