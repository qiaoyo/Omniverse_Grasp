# usd test about position, orientation, material and texture

# groundplane test about size , position, orientation, material and texture
import os,sys
import numpy as np
import omni
from omni.isaac.kit import SimulationApp
import yaml
import torch

def contact_report_event(contact_headers,contact_data):
    for contact_header in contact_headers:
            print("Got contact header type: " + str(contact_header.type))
            print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
            print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))
            print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0)))
            print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1)))
            print("StageId: " + str(contact_header.stage_id))
            print("Number of contacts: " + str(contact_header.num_contact_data))
            
            contact_data_offset = contact_header.contact_data_offset
            num_contact_data = contact_header.num_contact_data
            
            # for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
            #     print("Contact:")
            #     print("Contact position: " + str(contact_data[index].position))
            #     print("Contact normal: " + str(contact_data[index].normal))
            #     print("Contact impulse: " + str(contact_data[index].impulse))
            #     print("Contact separation: " + str(contact_data[index].separation))
            #     print("Contact faceIndex0: " + str(contact_data[index].face_index0))
            #     print("Contact faceIndex1: " + str(contact_data[index].face_index1))
            #     print("Contact material0: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material0)))
            #     print("Contact material1: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material1)))


if __name__=="__main__":


    CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

    # with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/assemble/config.yaml') as f:
    #     config=yaml.load(f)
    
    kit=SimulationApp(launch_config=CONFIG)

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
    ground_plane=GroundPlane(prim_path="/World/ground_plane",size=500)

    compoment_name='056_1'
    usd_folder='/home/pika/Desktop/assembled/056/_converted/'
    usd_path=usd_folder+compoment_name+'_STL.usd'
    print(usd_path)

    stage=omni.usd.get_context().get_stage()


    create_prim(
        prim_path="/World/Assemble_1",
        position=[0,0,0],
        # orientation=[0.7,0.7,0,0],
        scale=[0.1,0.1,0.1],
        usd_path=usd_path,
        semantic_label='056_1'
    )

    component_prim_1=stage.GetPrimAtPath("/World/Assemble_1")


    stage = omni.usd.get_context().get_stage()
    result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the prim
    cube_prim = stage.GetPrimAtPath("/Cube")
# Enable physics on prim
# If a tighter collision approximation is desired use convexDecomposition instead of convexHull
    utils.setRigidBody(cube_prim, "convexHull", False)


    # the second prim
    compoment_name='056_2'
    usd_folder='/home/pika/Desktop/assembled/056/_converted/'
    usd_path=usd_folder+compoment_name+'_STL.usd'
    print(usd_path)

    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,0.01)

    create_prim(
        prim_path="/World/Assemble_2",
        position=[0,0,0],
        # orientation=[0.7,0.7,0,0],
        scale=[1,1,1],
        usd_path=usd_path,
        semantic_label='056_2'
    )

    component_prim_2=stage.GetPrimAtPath("/World/Assemble_2")



    # xform=UsdGeom.Xformable(component_prim)
    # transform=xform.AddTransformOp()
    # mat=Gf.Matrix4d()
    # mat.SetTranslateOnly(Gf.Vec3d(10,1,1))
    # mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,1,0),290))
    # transform.Set(mat)

    utils.setRigidBody(component_prim_1,"convexHull",False)
    utils.setRigidBody(component_prim_2,"convexHull",False)

    # mass_api=UsdPhysics.MassAPI.Apply(component_prim)
    # mass_api.CreateMassAttr(10)
    # mass_api.CreateDensityAttr(100)

    # success,result=omni.kit.commands.execute(
    #     'CreateMdlMaterialPrimCommand',
    #     mtl_url='OmniPBR.mdl',
    #     mtl_name='OmniPBR',
    #     mtl_path='/World/Assemble_1/Looks/OmniPBR'
    # )

    success,result=omni.kit.commands.execute(
    'CreateMdlMaterialPrimCommand',
    mtl_url='/home/pika/Downloads/Metals/Aluminum_Anodized.mdl',
    mtl_name='Aluminum_Anodized',
    mtl_path='/World/Assemble_1/Looks/Aluminum'
    )

    mtl_prim=stage.GetPrimAtPath('/World/Assemble_1/Looks/Aluminum')
    
    # omni.usd.create_material_input(
    #     mtl_prim,
    #     "diffuse_texture",
    #     '/home/pika/Desktop/mtlandtexture/Ground/textures/mulch_norm.jpg',
    #     Sdf.ValueTypeNames.Asset
    # )


    shade=UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(component_prim_1).Bind(shade,UsdShade.Tokens.strongerThanDescendants)

    # contact_report_sub=get_physx_simulation_interface().subscribe_contact_report_events(contact_report_event)

    # contactReportAPI_1=PhysxSchema.PhysxContactReportAPI.Apply(component_prim_1)
    # contactReportAPI_2=PhysxSchema.PhysxContactReportAPI.Apply(component_prim_2)
    # contactReportAPI_1.CreateThresholdAttr().Set(200000)
    # contactReportAPI_2.CreateThresholdAttr().Set(200000)

    # # contactreport=ContactReportDirectAPIDemo().create(stage=stage)



    kit.update()
    kit.update()

    while True:
        kit.update()
