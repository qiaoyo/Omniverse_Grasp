import os,sys
import numpy as np
# import omni
# from omni.isaac.kit import SimulationApp
# import yaml
# import torch
# import time

# if __name__=="__main__":


#     CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

#     # with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/assemble/config.yaml') as f:
#     #     config=yaml.load(f)
    
#     simulation_app=SimulationApp(launch_config=CONFIG)

#     from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
#     from omni.isaac.core import World
#     from omni.isaac.core.objects import GroundPlane
#     from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface
#     import omni.usd
#     from omni.isaac.core.utils.prims import create_prim
#     from omni.isaac.core.utils.nucleus import get_assets_root_path
#     from omni.physx.scripts import utils
#     from omni.physx.scripts.physicsUtils import *
#     from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema
#     from omni.physx import get_physx_interface, get_physx_simulation_interface
#     from contactimmediate import ContactReportDirectAPIDemo
#     import omni.kit.commands
#     from omni.isaac.core.utils import prims
#     import omni.replicator.core as rep

#     my_world=World(stage_units_in_meters=0.01)
#     ground_plane=GroundPlane(prim_path="/World/ground_plane",size=500)



#     compoment_name='056_1'
#     usd_folder='/home/pika/Desktop/assembled/056/_converted/'
#     usd_path=usd_folder+compoment_name+'_STL.usd'
#     print(usd_path)

#     stage=omni.usd.get_context().get_stage()


#     create_prim(
#         prim_path="/World/Assemble_1",
#         position=[0,0,0],
#         # orientation=[0.7,0.7,0,0],
#         scale=[1,1,1],
#         usd_path=usd_path,
#         semantic_label='056_1'
#     )

#     component_prim_1=stage.GetPrimAtPath("/World/Assemble_1")
#     utils.setRigidBody(component_prim_1,"convexHull",False)


#     # camera = prims.create_prim(
#     #     prim_path="/World/Camera",
#     #     prim_type="Camera",
#     #     attributes={
#     #     "focusDistance": 1,
#     #     "focalLength": 24,
#     #      "horizontalAperture": 20.955,
#     #     "verticalAperture": 15.2908,
#     #     "clippingRange": (0.01, 1000000),
#     #     "clippingPlanes": np.array([1.0, 0.0, 1.0, 1.0]),
#     #     },
#     # )

#     rep_camera=rep.create.camera(
#         focus_distance=400.0, 
#         focal_length=24.0, 
#         clipping_range=(0.1, 10000000.0), 
#         name="Camera"
#     )
    

#     with rep_camera:
#         rep.modify.pose(
#             position=(100,50,300),
#             rotation=(0,-90,0)
#         )


#     writer=rep.WriterRegistry.get("BasicWriter")
#     output_directory = '/home/pika/Desktop/assembled/_output_headless'
#     print("Outputting data to ", output_directory)
#     writer.initialize(
#         output_dir=output_directory,
#         rgb=True,
#         # bounding_box_2d_tight=True,
#         # semantic_segmentation=True,
#         # instance_segmentation=True,
#         # distance_to_image_plane=True,
#         distance_to_camera=True,
#         # bounding_box_3d=True,
#         # occlusion=True,
#         # normals=True,
#     )

#     RESOLUTION=(CONFIG['width'],CONFIG['height'])
#     rep_product=rep.create.render_product(rep_camera,RESOLUTION)
#     writer.attach([rep_product])
#     # render_product=rep.create.render_product(rep_camera,resolution=(640,480))

    
#     # # rgb_data=rep.AnnotatorRegistry.get_annotator("rgb")
#     # # depth_data=rep.AnnotatorRegistry.get_annotator("distance_to_camera")
#     # # rgb_data.attach([render_product])
#     # # depth_data.attach([render_product])

#     for i in range(50):
#         rep.orchestrator.step()

#     while True:
#         simulation_app.update()

    
#     simulation_app.close()

    


if __name__=='__main__':
    data_path='/home/pika/Desktop/assembled/_output_headless/distance_to_camera_0000.npy'

    data=np.load(data_path)
    print(data.shape)

    print(data)