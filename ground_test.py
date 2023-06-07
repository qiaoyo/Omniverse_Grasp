# groundplane test about size , position, orientation, material and texture
import os,sys
import numpy as np
import omni
from omni.isaac.kit import SimulationApp
import yaml
if __name__=="__main__":


    CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

    with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/assemble/config.yaml') as f:
        config=yaml.load(f)
    
    kit=SimulationApp(launch_config=CONFIG)

    from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
    from omni.isaac.core import World
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface

    my_world=World(stage_units_in_meters=0.01)

    
    ground_plane=GroundPlane(
                prim_path=config['GroundPlane']['prim'],
                size=config['GroundPlane']['size'],
                z_position=config['GroundPlane']['z_position'],
                scale=np.array(config['GroundPlane']['scale']),
                #  color=np.array([1,0,0]),
                physics_material=PhysicsMaterial(
                                prim_path=config['GroundPlane']['PhysicsMaterial']['prim'],
                                dynamic_friction=config['GroundPlane']['PhysicsMaterial']['dynamic_friction'],
                                static_friction=config['GroundPlane']['PhysicsMaterial']['static_friction'],
                                restitution=config['GroundPlane']['PhysicsMaterial']['restitution']),
                visual_material=PreviewSurface(
                               prim_path=config['GroundPlane']['PreviewSurface']['prim'], 
                               color=np.array(config['GroundPlane']['PreviewSurface']['color']),
                            #    shader=shader,
                               roughness=config['GroundPlane']['PreviewSurface']['roughness'],
                               metallic=config['GroundPlane']['PreviewSurface']['metallic'])
                               )
    
    kit.update()
    kit.update()

    while True:
        kit.update()
