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

def create_background(Config_yaml):
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema   
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface
    from omni.isaac.core.materials import OmniPBR

    stage=omni.usd.get_context().get_stage()

    visual_material=OmniPBR(
                        prim_path='/World/Looks/Visual_material',
                        texture_path='/home/pika/Desktop/mtlandtexture/Ground/textures/cobblestone_medieval_diff.jpg',
                        texture_scale=np.array([0.01,0.010,0.010])
                )
    visual_material.set_metallic_constant(0.3)
    visual_material.set_reflection_roughness(0.1)

    physics_material=PhysicsMaterial(
                                prim_path=Config_yaml['GroundPlane']['PhysicsMaterial']['prim'],
                                dynamic_friction=Config_yaml['GroundPlane']['PhysicsMaterial']['dynamic_friction'],
                                static_friction=Config_yaml['GroundPlane']['PhysicsMaterial']['static_friction'],
                                restitution=Config_yaml['GroundPlane']['PhysicsMaterial']['restitution'])
    
    ground_plane=GroundPlane(
                prim_path=Config_yaml['GroundPlane']['prim'],
                size=Config_yaml['GroundPlane']['size'],
                z_position=Config_yaml['GroundPlane']['z_position'],
                scale=np.array(Config_yaml['GroundPlane']['scale']),
                
                color=np.array([1,0,0]),
                physics_material=physics_material,
                visual_material=visual_material
                               )
    
    ground_prim=stage.GetPrimAtPath(Config_yaml['GroundPlane']['prim'])

    # success,result=omni.kit.commands.execute(
    #     'CreateMdlMaterialPrimCommand',
    #     mtl_url='/home/pika/Desktop/mtlandtexture/Ground/Mulch.mdl',
    #     mtl_name='Mulch',
    #     mtl_path='/World/GroundPlane/Looks/Mulch'
    # )

    # mtl_prim=stage.GetPrimAtPath('/World/GroundPlane/Looks/Mulch')

#     omni.usd.create_material_input(
#         mtl_prim,
#         "endless_texture",
#         '/home/pika/Desktop/mtlandtexture/Ground/textures/mulch_norm.jpg',
#         Sdf.ValueTypeNames.Asset,
# )

    # shade=UsdShade.Material(mtl_prim)
    # UsdShade.MaterialBindingAPI(ground_prim).Bind(shade,UsdShade.Tokens.strongerThanDescendants)

    return ground_plane

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
    from omni.physx.scripts import utils
    from omni.isaac.core.utils import prims
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.materials import OmniPBR
    from omni.isaac.core.prims import GeometryPrim, XFormPrim
    stage=omni.usd.get_context().get_stage()

    visual_material=OmniPBR(
                        prim_path='/World/GroundPlane/Looks/visual_material',
                        texture_path='/opt/nvidia/mdl/vMaterials_2/Metal/textures/iron_pitted_steel_heat_diff.jpg',
                        texture_scale=np.array([1,1,1])
                )
    visual_material.set_metallic_constant(0.3)
    visual_material.set_reflection_roughness(0.5)


    for part in total_parts:
        part_name=part
        usd_path=Config_yaml['DataPath']['PartsPath']+part[0:3]+'/_converted/'+part
        
        position,orientation=random_six_dof(Config_yaml)
        prim_path="/World/Parts_"+part[:-4]
        print(prim_path)
        semantic_label=part[:-4]
        part_prim=prims.create_prim(
            prim_path=prim_path,
            position=position,
            orientation=orientation,
            scale=[0.01,0.01,0.01],
            usd_path=usd_path,
            semantic_label=semantic_label
        )
        print(part_prim)
        part_prim=stage.GetPrimAtPath(prim_path)
        xformprim=XFormPrim(
            prim_path=prim_path,name='part'+semantic_label,position=position,orientation=orientation,scale=np.array([0.1,0.1,0.1]),visible=True
        )
        xformprim.apply_visual_material(visual_material=visual_material)
        utils.setRigidBody(part_prim,"convexHull",False)



        # part_rigid_prim=RigidPrim(
        #     prim_path=str(part_prim.GetPrimPath()),
        #     name="Parts_"+part[:-8]
        # )
        # part_rigid_prim.enable_rigid_body_physics()
        # world.scene.add(part_rigid_prim)

def create_render_camera(Config_yaml):
    import omni.replicator.core as rep
    camera_num=Config_yaml['Camera']['num']
    camera_position=np.zeros((camera_num,3))
    camera_orientation=np.zeros((camera_num,3))


    writer=rep.WriterRegistry.get("BasicWriter")
    output_directory='/home/pika/Desktop/assembled/_output_headless'
    
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        distance_to_camera=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        bounding_box_3d=True,
        normals=True
    )

    rep_product_list=[]
    for i in range(camera_num):
        rep_camera=rep.create.camera(
            focus_distance=Config_yaml['Camera']['focus_distance'],
            focal_length=Config_yaml['Camera']['focal_length'],
            name='Camera'+str(i+1),
        )

        with rep_camera:
            rep.modify.pose(
                position= (0,0,300),#camera_position[i],
                rotation=(0,-90,0),#camera_orientation[i]
            )

        RESOLUTION=(Config_yaml['WorldConfig']['width'],Config_yaml['WorldConfig']['height'])
        rep_product=rep.create.render_product(rep_camera,resolution=RESOLUTION)
        rep_product_list.append(rep_product)
    writer.attach(rep_product_list)

# def register_random():
    
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


async def pause_sim(timeline,task):
    done ,pending = await asyncio.wait({task})

    if task in done:
        print("Waited for 5 seconds")
        timeline.pause()

def main():

    with open('./Assemble/config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)

    kit=SimulationApp(launch_config=Config_yaml['WorldConfig'])

    from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
    from omni.isaac.core import World
    import omni.usd
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from contactimmediate import ContactReportDirectAPIDemo
    import omni.kit.commands
    import omni.replicator.core as rep

    my_world=World(stage_units_in_meters=0.01)
    stage=omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)


    # ground=my_world.scene.add_ground_plane()
    # PhysicsSchemaTools.addGroundPlane(stage, "/groundPlane", "Z", 2500, Gf.Vec3f(0, 0, -100), Gf.Vec3f([0.5,0.5,0.5]))


    create_background(Config_yaml)

    target_procedure='056'  #0~6

    total_parts_num=10  

    procedure_parts_num=len(os.listdir(Config_yaml['DataPath']['PartsPath']+target_procedure+'/_converted/'))

    random_parts_num=total_parts_num-procedure_parts_num

    procedure_parts=os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/')

    random_parts=select_random_parts(Config_yaml,random_parts_num,target_procedure)

    total_parts=procedure_parts+random_parts

    create_component(Config_yaml,total_parts)


    # timeline=omni.timeline.get_timeline_interface()

    # create_render_camera(Config_yaml)


    # my_world.start_simulation()
    # timeline.set_start_time(0.0)
    # timeline.set_end_time(5.0)

    # timeline.set_end_time(10)
    # timeline.play()

    # task=asyncio.ensure_future(omni.kit.app.get_app().next_update_async())
    # asyncio.ensure_future(pause_sim(timeline,task))

    # timeline.pause()
    # my_world.reset()


    # num_sim_steps=Config_yaml['num_sim_steps']
    # print(num_sim_steps)
    # for i in range(num_sim_steps):
    #     my_world.step(render=False)

    
    # for i in range(20):
    #     rep.orchestrator.step()

    # kit.update()

    # timeline.play()
    k=0
    kit.update()
    kit.update()
    kit.update()
    while True:
        # k+=1
        # my_world.step()
        # rep.orchestrator.step(pause_timeline=False)
        # timeline.play()
        # if k ==10:
        #     break
        kit.update()
    # while True:
    #     kit.update()
    # kit.close()

def test_6dof_random_parts():
    with open('./config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)
    position,orientation=random_six_dof(Config_yaml=Config_yaml)

    print(position)
    print(orientation)

    target_procedure=1  #0~6

    total_parts_num=10  

    procedure_parts_num=len(os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/'))

    random_parts_num=total_parts_num-procedure_parts_num

    procedure_parts=os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/')

    random_parts=select_random_parts(Config_yaml,random_parts_num,target_procedure)

    total_parts=procedure_parts+random_parts

    print(total_parts)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.print_exc())
    finally:
        # kit.close()
        print('finished')
