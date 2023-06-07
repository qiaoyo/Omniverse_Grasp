import omni
from omni.isaac.kit import SimulationApp
import os
import yaml
import numpy as np
import random
import cv2
from scipy.spatial.transform import Rotation as R

def create_background(Config_yaml):   
    ground_plane=GroundPlane(
                prim_path=Config_yaml['GroundPlane']['prim'],
                size=Config_yaml['GroundPlane']['size'],
                z_position=Config_yaml['GroundPlane']['z_position'],
                scale=np.array(Config_yaml['GroundPlane']['scale']),
                #  color=np.array([1,0,0]),
                physics_material=PhysicsMaterial(
                                prim_path=Config_yaml['GroundPlane']['PhysicsMaterial']['prim'],
                                dynamic_friction=Config_yaml['GroundPlane']['PhysicsMaterial']['dynamic_friction'],
                                static_friction=Config_yaml['GroundPlane']['PhysicsMaterial']['static_friction'],
                                restitution=Config_yaml['GroundPlane']['PhysicsMaterial']['restitution']),
                visual_material=PreviewSurface(
                               prim_path=Config_yaml['GroundPlane']['PreviewSurface']['prim'], 
                               color=np.array(Config_yaml['GroundPlane']['PreviewSurface']['color']),
                            #    shader=shader,
                               roughness=Config_yaml['GroundPlane']['PreviewSurface']['roughness'],
                               metallic=Config_yaml['GroundPlane']['PreviewSurface']['metallic'])
                               )
    return True

def select_random_parts(Config_yaml,random_parts_num,target_procedure):
    PartsPath=Config_yaml['DataPath']['PartsPath']
    parts=os.listdir(PartsPath)
    parts.remove(str(target_procedure))

    to_be_randomed=[]
    for part in parts:
        part_path=Config_yaml['DataPath']['PartsPath']+part+'/_converted/'
        to_be_randomed.extend(os.listdir(part_path))

    random.shuffle(to_be_randomed)

    return to_be_randomed[:random_parts_num]

def create_component(Config_yaml,total_parts):

    stage=omni.usd.get_context().get_stage()
    for part in total_parts:
        part_name=part
        usd_path=Config_yaml['DataPath']['PartsPath']+part[0:3]+'/_converted/'+part
        
        position,orientation=random_six_dof(Config_yaml)
        prim_path="/World/Parts_"+part[:-8]
        print(prim_path)
        semantic_label=part[:-8]
        create_prim(
            prim_path=prim_path,
            position=position,
            orientation=orientation,
            scale=[1,1,1],
            usd_path=usd_path,
            # semantic_label=semantic_label
        )
        part_prim=stage.GetPrimAtPath(prim_path)
        utils.setRigidBody(part_prim,"convexHull",False)

def random_six_dof(Config_yaml):
    position=np.zeros(3)
    # orientation=np.zeros(4)
    x_min=Config_yaml['Component']['x_min']
    x_max=Config_yaml['Component']['x_max']
    y_min=Config_yaml['Component']['y_min']
    y_max=Config_yaml['Component']['y_max']
    z_min=Config_yaml['Component']['z_min']
    z_max=Config_yaml['Component']['z_max']
    position[0]=random.uniform(x_min,x_max)
    position[1]=random.uniform(y_min,y_max)
    position[2]=random.uniform(z_min,z_max)

    rot_vector=np.random.randn(3)
    rot_vector=rot_vector/np.linalg.norm(rot_vector)
    rot_angle=random.uniform(0,2*np.pi)
    rot_vector=rot_angle*rot_vector
    rot_matrix=cv2.Rodrigues(rot_vector)[0]
    R_rot=R.from_matrix(rot_matrix)
    orientation=np.array(R_rot.as_quat())

    return position,orientation

# def collision_check():


if __name__=='__main__':

    with open('./Assemble/config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)

    kit=SimulationApp(launch_config=Config_yaml['WorldConfig'])

    from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
    from omni.isaac.core import World
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface
    import omni.usd
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    from omni.physx.scripts.physicsUtils import *
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from contactimmediate import ContactReportDirectAPIDemo
    import omni.kit.commands

    my_world=World(stage_units_in_meters=0.01)
    stage=omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,0.01)

    create_background(Config_yaml)

    target_procedure='056'  #0~6

    total_parts_num=12  

    procedure_parts_num=len(os.listdir(Config_yaml['DataPath']['PartsPath']+target_procedure+'/_converted/'))

    random_parts_num=total_parts_num-procedure_parts_num

    procedure_parts=os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/')

    random_parts=select_random_parts(Config_yaml,random_parts_num,target_procedure)

    total_parts=procedure_parts+random_parts

    create_component(Config_yaml,total_parts)

    # collision_check()

    

    # omni.timeline.get_timeline_interface().play()

    kit.update()
    kit.update()

    while True:
        kit.update()
    kit.close()

# if __name__=='__main__':
#     with open('./config.yaml') as Config_file:
#         Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)
#     position,orientation=random_six_dof(Config_yaml=Config_yaml)

#     print(position)
#     print(orientation)

#     target_procedure=1  #0~6

#     total_parts_num=10  

#     procedure_parts_num=len(os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/'))

#     random_parts_num=total_parts_num-procedure_parts_num

#     procedure_parts=os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/')

#     random_parts=select_random_parts(Config_yaml,random_parts_num,target_procedure)

#     total_parts=procedure_parts+random_parts

#     print(total_parts)
