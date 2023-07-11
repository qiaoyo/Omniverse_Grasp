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

def random_ground(Config_yaml):
    from omni.isaac.core.materials import OmniPBR

    ground_path=Config_yaml['DataPath']['MdlPath']
    Ground_type=['Ground','Concrete','Wood',"Leather"]
    random.shuffle(Ground_type)
    ground_path=ground_path+Ground_type[0]
    ground_name=os.listdir(ground_path)
    random.shuffle(ground_name)
    ground=ground_name[0]
    ground_datum=os.listdir(ground_path+'/'+ground)
    if ground_datum[0][-4:]=='.npy':
        texture=ground_datum[1]
        roughness=np.load(ground_path+'/'+ground+'/roughness.npy')
    else:
        texture=ground_datum[0]
        roughness=np.load(ground_path+'/'+ground+'/roughness.npy')

    visual_material=OmniPBR(
                        prim_path='/World/Looks/Visual_material',
                        texture_path=ground_path+'/'+ground+'/'+texture,
                        texture_scale=np.array([0.01,0.01,0.01])
                )

    visual_material.set_reflection_roughness(float(roughness))

    return visual_material

def random_mtl(Config_yaml):
    mdl_path=Config_yaml['DataPath']['MdlPath']+'Metal'
    mdl_full=os.listdir(mdl_path)

    mdl=[]
    for file in mdl_full:
        if file[-4:]=='.mdl':
            mdl.append(file)
    random.shuffle(mdl)
    return mdl_path+'/'+mdl[0],mdl[0][:-4]

def create_background(Config_yaml):
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema   
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface

    stage=omni.usd.get_context().get_stage()

    visual_material=random_ground(Config_yaml)
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
                
                color=np.array([0.5,0.5,0.5]),
                physics_material=physics_material,
                visual_material=visual_material
                               )
    
    ground_prim=stage.GetPrimAtPath(Config_yaml['GroundPlane']['prim'])

    return ground_prim

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
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema 
    from omni.isaac.core.utils import prims
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.materials import OmniPBR
    from omni.isaac.core.prims import GeometryPrim, XFormPrim

    stage=omni.usd.get_context().get_stage()
    PartsPath=Config_yaml['DataPath']['PartsPath']
    i=1
    random.shuffle(total_parts)

    for part in total_parts:
        part_name=part
        usd_path=Config_yaml['DataPath']['PartsPath']+part[0:3]+'/_converted/'+part
        
        position,orientation=random_six_dof(Config_yaml)
        position[2]=i*10
        i=i+1
        prim_path="/World/Parts_"+part[:-4]
        print(prim_path)
        semantic_label=part[:-4]
        part_scale=np.load(PartsPath+part[0:3]+'/scale.npy')[int(part[4])-1]
        part_scale=float(np.random.normal(part_scale,0.02,1))
        part_prim=prims.create_prim(
            prim_path=prim_path,
            position=position,
            orientation=orientation,
            scale=[part_scale,part_scale,part_scale],
            usd_path=usd_path,
            semantic_label=semantic_label
        )
        # print(part_prim)

        mdl_url,mdl_name=random_mtl(Config_yaml)
        success,result=omni.kit.commands.execute(
            'CreateMdlMaterialPrimCommand',
            mtl_url=mdl_url,
            mtl_name=mdl_name,
            mtl_path=prim_path+'/Looks/'+mdl_name
        )

        mtl_prim=stage.GetPrimAtPath(prim_path+'/Looks/'+mdl_name)
    
        shade=UsdShade.Material(mtl_prim)
        UsdShade.MaterialBindingAPI(part_prim).Bind(shade,UsdShade.Tokens.strongerThanDescendants)

        # xformprim=XFormPrim(
        #     prim_path=prim_path,name='part'+semantic_label,position=position,orientation=orientation,scale=np.array([0.1,0.1,0.1]),visible=True
        # )
        # xformprim.apply_visual_material(visual_material=visual_material)
        utils.setRigidBody(part_prim,"convexHull",False)

def random_camera_pose(Config_yaml):
    radius_min=Config_yaml['Random']['Camera']['radius_min']
    radius_max=Config_yaml['Random']['Camera']['radius_max']

    look_at_min=Config_yaml['Random']['Camera']['look_at_min']
    look_at_max=Config_yaml['Random']['Camera']['look_at_max']

    x_min=Config_yaml['Random']['Camera']['x_min_norm']
    x_max=Config_yaml['Random']['Camera']['x_max_norm']
    y_min=Config_yaml['Random']['Camera']['y_min_norm']
    y_max=Config_yaml['Random']['Camera']['y_max_norm']

    render_steps=Config_yaml['Renderer']['Render_steps']

    position=np.zeros((render_steps,3))
    look_at=np.zeros((render_steps,3))

    for i in range(render_steps):
        r=random.uniform(radius_min,radius_max)
        x=random.uniform(x_min,x_max)
        y=random.uniform(y_min,y_max)
        z=math.sqrt(1-x**2-y**2)
        vector=np.array([x,y,z])

        look_at[i]=np.array([
            random.uniform(look_at_min,look_at_max),
            random.uniform(look_at_min,look_at_max),
            random.uniform(look_at_min,look_at_max),
        ])

        position[i]=look_at[i]+r*vector

    position_list=[]
    look_at_list=[]

    for i in range(render_steps):
        position_tuple=(float(position[i][0]),float(position[i][1]),float(position[i][2]))
        look_at_tuple=(float(look_at[i][0]),float(look_at[i][1]),float(look_at[i][2]))
        
        position_list.append(position_tuple)
        look_at_list.append(look_at_tuple)


    return position_list,look_at_list

def register_camera(Config_yaml,args):
    import omni.replicator.core as rep
    camera_num=Config_yaml['Camera']['num']

    writer=rep.WriterRegistry.get("BasicWriter")
    output_directory=Config_yaml['DataPath']['OutPath']+args.path+'/'
    
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        distance_to_camera=True,
        # distance_to_image_plane=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        bounding_box_3d=True,
        occlusion=True,
        # pointcloud=True,
        normals=True
    )

    rep_product_list=[]
    rep_camera_list=[]
    for i in range(camera_num):
        rep_camera=rep.create.camera(
            focus_distance=Config_yaml['Camera']['focus_distance'],
            focal_length=Config_yaml['Camera']['focal_length'],
            name='Camera'+str(i+1),
        )
        rep_camera_list.append(rep_camera)

        RESOLUTION=(Config_yaml['Camera']['width'],Config_yaml['Camera']['height'])
        rep_product=rep.create.render_product(rep_camera,resolution=RESOLUTION)
        rep_product_list.append(rep_product)

    writer.attach(rep_product_list)
    return rep_camera_list

def register_light(Config_yaml):
    import omni.replicator.core as rep
    x_y_min=Config_yaml['Random']['Light']['Position']['x_y_min']
    x_y_max=Config_yaml['Random']['Light']['Position']['x_y_max']
    z_min=Config_yaml['Random']['Light']['Position']['z_min']
    z_max=Config_yaml['Random']['Light']['Position']['z_max']
    scale_min=Config_yaml['Random']['Light']['Scale']['min']
    scale_max=Config_yaml['Random']['Light']['Scale']['max']
    def sphere_lights():
        lights=rep.create.light(
            light_type=Config_yaml['Random']['Light']['Type'],
            intensity=rep.distribution.normal(Config_yaml['Random']['Light']['Intensity']['Mean'],Config_yaml['Random']['Light']['Intensity']['Delta']),
            position=rep.distribution.uniform((x_y_min,x_y_min,z_min), (x_y_max,x_y_max,z_max)),
            scale=rep.distribution.uniform((scale_min,scale_min,scale_min),(scale_max,scale_max,scale_max)),
        )
        return lights.node
    rep.randomizer.register(sphere_lights)

def register_camera_look_at(Config_yaml):
    from omni.isaac.core.utils.prims import create_prim,get_prim_path
    import omni.replicator.core as rep

    camera_num=Config_yaml['Camera']['num']
    camera_look_at_node_list=[]
    camera_look_at_prim_list=[]
    for i in range(camera_num):
        camera_look_at_prim=create_prim(
            prim_path='/World/Look_at'+str(i),
        )
        camera_look_at_node=rep.get.prim_at_path(str(camera_look_at_prim.GetPath()))

        camera_look_at_node_list.append(camera_look_at_node)
        camera_look_at_prim_list.append(get_prim_path(camera_look_at_prim))
    return camera_look_at_node_list,camera_look_at_prim_list

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
    # position[2]=random.uniform(z_min,z_max)

    rot_vector=np.random.randn(3)
    rot_vector=rot_vector/np.linalg.norm(rot_vector)
    rot_angle=random.uniform(0,2*np.pi)
    rot_vector=rot_angle*rot_vector
    rot_matrix=cv2.Rodrigues(rot_vector)[0]
    R_rot=R.from_matrix(rot_matrix)
    orientation=np.array(R_rot.as_quat())

    # print(position)
    return position,orientation

def save_6dof(Config_yaml,args,total_parts):
    from omni.isaac.core.utils.prims import get_prim_at_path

    total_parts_file=Config_yaml['DataPath']['OutPath']+args.path+'/Total_Parts.txt'
    f=open(total_parts_file,'w')

    pos=[]
    rot_im=[]
    rot_rel=[]
    for part in total_parts:
        f.writelines(part[:-4]+'\n')
        prim_path='/World/Parts_'+part[:-4]
        part_prim=get_prim_at_path(prim_path)
        part_tf=omni.usd.get_world_transform_matrix(part_prim)
        part_pos=part_tf.ExtractTranslation()
        part_rot=part_tf.ExtractRotation().GetQuaternion()
        pos.append(part_pos)
        rot_im.append(part_rot.GetReal())
        rot_rel.append(part_rot.GetImaginary())
    # print(pos,rot_im,rot_rel)

    f.close()
    np.save(Config_yaml['DataPath']['OutPath']+args.path+'/Parts_Pos.npy',np.array(pos))
    np.save(Config_yaml['DataPath']['OutPath']+args.path+'/Parts_Rot_im.npy',np.array(rot_im))
    np.save(Config_yaml['DataPath']['OutPath']+args.path+'/Parts_Rot_rel.npy',np.array(rot_rel))
    
def get_camera_6dof(Config_yaml):
    from omni.isaac.core.utils.prims import get_prim_at_path
    Camera_num=Config_yaml['Camera']['num']

    camera_pos=[]
    camera_rot_im=[]
    camera_rot_rel=[]
    for i in range(Camera_num):
        prim_path='/Replicator/Camera'+str(i+1)+'_Xform'
        camera_prim=get_prim_at_path(prim_path)
        camera_tf=omni.usd.get_world_transform_matrix(camera_prim)
        camera_pos.append(camera_tf.ExtractTranslation())
        camera_rot_im.append(camera_tf.ExtractRotation().GetQuaternion().GetReal())
        camera_rot_rel.append(camera_tf.ExtractRotation().GetQuaternion().GetImaginary())
    return camera_pos,camera_rot_im,camera_rot_rel

async def pause_sim(timeline,task):
    done ,pending = await asyncio.wait({task})

    if task in done:
        print("Waited for 5 seconds")
        timeline.pause()

def main():

    with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/Assemble/config.yaml') as Config_file:
        Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)

    kit=SimulationApp(launch_config=Config_yaml['WorldConfig'])

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

    parser=argparse.ArgumentParser(description='Parser')
    parser.add_argument('-t','--target',type=str,help='Target procedure')
    parser.add_argument('-n','--number',type=int,help='Number of the total parts')
    parser.add_argument('-p','--path',type=str,help='Path to save the data')
    
    args=parser.parse_args()
    
    # Create World and get stage
    my_world=World(stage_units_in_meters=0.01)
    my_stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(my_stage,UsdGeom.Tokens.z)


    success,result=omni.kit.commands.execute(
        'SetLightingMenuModeCommand',
        lighting_mode='Default',
    )

    # Create BackGround and Register Light Randomizer
    ground_prim=create_background(Config_yaml)
    register_light(Config_yaml)
    
    # Create Camera and Register Camera_look_at Randomizer
    rep_camera_list=register_camera(Config_yaml,args)
    camera_look_at_node_list,camera_look_at_prim_list=register_camera_look_at(Config_yaml)
    
    # Check if the out_dir exists or not
    # output_directory=Config_yaml['DataPath']['OutPath']
    
    # if os.path.exists(output_directory):
    #     shutil.rmtree(output_directory)

    output_directory=Config_yaml['DataPath']['OutPath']+args.path+'/'

    # Set the target Procedure and Generate the total parts
    target_procedure=args.target #0~6
    total_parts_num=args.number
    procedure_parts_num=len(os.listdir(Config_yaml['DataPath']['PartsPath']+target_procedure+'/_converted/'))

    random_parts_num=total_parts_num-procedure_parts_num

    procedure_parts=os.listdir(Config_yaml['DataPath']['PartsPath']+str(target_procedure)+'/_converted/')

    random_parts=select_random_parts(Config_yaml,random_parts_num,target_procedure)

    total_parts=procedure_parts+random_parts

    # Create the prim of parts and random the material
    create_component(Config_yaml,total_parts)

    # Save the Original Stage
    # default_server = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
    # omni.usd.get_context().save_as_stage(default_server + "/Users/test/saved.usd", None)
    # omni.usd.get_context().save_stage(None)
    save_stage(output_directory+'Original.usd',False)

    # Get the timeline and Play
    timeline=omni.timeline.get_timeline_interface()

    timeline.play()
    step=0
    kit.update()
    kit.update()
    
    simulation_steps=Config_yaml['Renderer']['Simulation_steps']
    render_steps=Config_yaml['Renderer']['Render_steps']

    # The Falling Simulation Process 
    while step<=simulation_steps:
        step+=1
        my_world.step(render=False)

    # Save the Stage After Falling
    save_stage(output_directory+'After_Falling.usd',False)
    save_6dof(Config_yaml,args,total_parts)
    # The Randomization Trigger Definition
    camera_num=Config_yaml['Camera']['num']
    with rep.trigger.on_frame(num_frames=render_steps):
        
        # The Randomization of Sphere Light
        rep.randomizer.sphere_lights()
        
        i=0
        for i in range(camera_num):
            # Random the Camera Position and Look_at
            camera_position,camera_look_at=random_camera_pose(Config_yaml)
            
            # Save the Camera Position and Look_at 
            np.save(output_directory+'Camera_'+str(i)+'_Position.npy',np.array(camera_position))
            np.save(output_directory+'Camera_'+str(i)+'_Look_at.npy',np.array(camera_look_at))

            # Modify the Camera Position ReplicatorItem
            with camera_look_at_node_list[i]:
                rep.modify.pose(
                    position=rep.distribution.sequence(camera_look_at)
                )
            
            # Modify the Camera Look_at ReplicatorItem
            with rep_camera_list[i]:
                rep.modify.pose(
                    position= rep.distribution.sequence(camera_position),
                    look_at= rep.distribution.sequence(camera_look_at_prim_list),
                )

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        kit.update()

    # The Render Process
    step=0

    saved_camera_pos=[]
    saved_camera_rot_im=[]
    saved_camera_rot_rel=[]
    while step<=render_steps:
        step+=1
        rep.orchestrator.step()
        camera_pos,camera_rot_im,camera_rot_rel=get_camera_6dof(Config_yaml)
        saved_camera_pos.append(camera_pos)
        saved_camera_rot_im.append(camera_rot_im)
        saved_camera_rot_rel.append(camera_rot_rel)
        kit.update()
    # print(saved_camera_pos,saved_camera_rot_im,saved_camera_rot_rel)
    np.save(output_directory+'Camera_Pos.npy',np.array(saved_camera_pos))
    np.save(output_directory+'Camera_Rot_im.npy',np.array(saved_camera_rot_im))
    np.save(output_directory+'Camera_Rot_rel.npy',np.array(saved_camera_rot_rel))

    # # Wait until stopped

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()

    # Pause the Timeline
    timeline.pause()

    # Keep the Simulation App
    # while True:
    #     kit.update()

    # Close the Simulation App
    kit.close()

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
