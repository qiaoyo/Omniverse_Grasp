import omni
from omni.isaac.kit import SimulationApp
import os
import yaml
import numpy as np
import random
import cv2
import time
from scipy.spatial.transform import Rotation as R
import asyncio
import math
import shutil
from quarternion import *
import argparse

# np.set_printoptions(precision=2, suppress=False)


if __name__=='__main__':

    with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/Omniverse_Grasp/config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)
    CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

    # kit=SimulationApp(launch_config=Config_yaml['WorldConfig'])
    kit=SimulationApp(launch_config=CONFIG)

    from omni.isaac.core.utils.stage import get_current_stage,get_stage_units,save_stage
    from omni.isaac.core import World
    import omni.usd
    from omni.isaac.core.utils.prims import create_prim,get_prim_at_path
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from contactimmediate import ContactReportDirectAPIDemo
    import omni.kit.commands
    import omni.replicator.core as rep
    import carb
    # parser=argparse.ArgumentParser(description='Parser')
    # parser.add_argument('-t','--target',type=str,help='Target procedure')
    # parser.add_argument('-n','--number',type=int,help='Number of the total parts')
    # parser.add_argument('-p','--path',type=str,help='Path to save the data')

    # args=parser.parse_args()

    my_world=World(stage_units_in_meters=0.01)
    my_stage=omni.usd.get_context().get_stage()

    total_parts=['001_1.usd']

    PartsPath=Config_yaml['DataPath']['PartsPath']
    i=1
    random.shuffle(total_parts)

# '''
#     scale!!!
#     scale!!!
#     scale!!
#     remember to use the SRT to get the 6dof pose and scale !!!
# '''
    for part in total_parts:
        part_name=part
        usd_path=Config_yaml['DataPath']['PartsPath']+part[0:3]+'/_converted/'+part
        print(usd_path)
        # position,orientation=random_six_dof(Config_yaml)
        # # print(part_name,position,orientation)
        # position[2]=i*10
        i=i+1
        prim_path="/World/Parts_"+part[:-4]
        # print(prim_path)
        semantic_label=part[:-4]
        part_scale=np.load(PartsPath+part[0:3]+'/scale.npy')[int(part[4])-1]
        part_scale=float(np.random.normal(part_scale,0.02,1))
        # part_scale=1
        print(part_scale)
        create_prim(
            prim_path=prim_path,
            position=[10,1,1],
            orientation=[0.8191520481182802, 0.12516463863621627, 0.25032927727243254, 0.5006585545448652],
            # position=position,
            # orientation=orientation,
            scale=[part_scale,part_scale,part_scale],
            usd_path=usd_path,
            semantic_label=semantic_label
        )

        part_prim=get_prim_at_path(prim_path)
        scale, rotation, rotation_order, translation=omni.usd.utils.get_local_transform_SRT(part_prim)
        print(part,scale,translation,rotation,euler2quaternion(rotation))

        # print(part_prim)

        # mdl_url,mdl_name=random_mtl(Config_yaml)
        # success,result=omni.kit.commands.execute(
        #     'CreateMdlMaterialPrimCommand',
        #     mtl_url=mdl_url,
        #     mtl_name=mdl_name,
        #     mtl_path=prim_path+'/Looks/'+mdl_name
        # )

        # mtl_prim=stage.GetPrimAtPath(prim_path+'/Looks/'+mdl_name)

        # shade=UsdShade.Material(mtl_prim)
        # UsdShade.MaterialBindingAPI(part_prim).Bind(shade,UsdShade.Tokens.strongerThanDescendants)

        # utils.setRigidBody(part_prim,"convexHull",False)

    # timeline=omni.timeline.get_timeline_interface()
    # timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

    while True:
        kit.update()



# if __name__=='__main__':
#     main()
# #     try:
# #         main()
#     except Exception as e:
#         import traceback
#         print(traceback.print_exc())
#     finally:
#         # kit.close()
#         print('finished')
