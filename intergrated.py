import os
import omni

from omni.isaac.kit import SimulationApp
import torch

if __name__=='__main__':

    CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

    kit =SimulationApp(launch_config=CONFIG)
    
    usd_folder='/home/pika/Desktop/assembled/56/_converted/'

    from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
    from omni.isaac.core import World
    from omni.isaac.core.objects import GroundPlane

    my_world=World(stage_units_in_meters=0.01)
    ground_plane=GroundPlane(prim_path="/World/ground_plane",size=1000)
    stage_unit=get_stage_units()
    print("stage unit:",stage_unit)
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom

    stage=omni.usd.get_context().get_stage()
    physic_scene=UsdPhysics.Scene.Define(stage,Sdf.Path("/World/physicsScene"))
    physic_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1))
    physic_scene.CreateGravityMagnitudeAttr().Set(981.0)

    # PhysicsSchemaTools.addGroundPlane(stage,"/World/groundPlane","Z",5000,Gf.Vec3f(0,0,0),Gf.Vec3f(1))


    part_name='056_1'
    usd_path=usd_folder+part_name+'.usd'
    print(usd_path)
    device='cuda'

    import omni.usd

    # stage.DefinePrim('/World/'+part_name)

    from omni.isaac.core.utils.prims import create_prim

    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils

    kit.update()
    kit.update()

    create_prim(
        prim_path="/World/Assemble_1",
        position=torch.tensor([0,0,10],device=device).cpu(),
        # translation=torch.tensor([0,0,0],device=device).cpu(),
        # orientation=torch.tensor([0.7,0.7,0,0],device=device).cpu(),
        scale=[1,1,1],
        usd_path=usd_path,
        semantic_label='056_1'
    )

    prim=stage.GetPrimAtPath("/World/Assemble_1/node_/mesh_")
    prim=stage.GetPrimAtPath("/World/Assemble_1")
    utils.setRigidBody(prim,"convexHull",False)
    mass_api=UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(10)
    mass_api.CreateDensityAttr(100)


    mtl_created_list=[]
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        mtl_created_list=mtl_created_list,
    )


    mtl_prim=stage.GetPrimAtPath(mtl_created_list[0])
    
    print(mtl_created_list,mtl_prim)

    omni.usd.create_material_input(mtl_prim,"glass_color",Gf.Vec3f(0,1,0),Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(mtl_prim,"glass_ior",1.0,Sdf.ValueTypeNames.Float)
    print(mtl_prim)
    shade=UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(prim).Bind(shade,UsdShade.Tokens.strongerThanDescendants)


    # # add a transform to a prim
    # xform=UsdGeom.Xformable(prim)
    # transform=xform.AddTransformOp()
    # mat=Gf.Matrix4d()
    # mat.SetTranslateOnly(Gf.Vec3d(10,1,1))
    # mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,1,0),290))
    # transform.Set(mat)
    

    while kit.is_running():

        kit.update()


    