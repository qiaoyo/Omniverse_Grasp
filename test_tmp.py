

from omni.isaac.kit import SimulationApp
import os

CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1024, "height": 1024, "num_frames": 10}
simulation_app = SimulationApp(launch_config=CONFIG)

ENV_URL = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
FORKLIFT_URL = "/Isaac/Props/Forklift/forklift.usd"
PALLET_URL = "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd"
CARDBOX_URL = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd"
CONE_URL = "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd"
SCOPE_NAME = "/MyScope"


import carb
import random
import math
import numpy as np
from pxr import UsdGeom, Usd, Gf, UsdPhysics, PhysxSchema

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, lookat_to_quatf
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache

import omni.replicator.core as rep

def simulate_falling_objects(num_sim_steps=20, num_boxes=8):
    # Create a simulation ready world
    from omni.isaac.core.objects import GroundPlane
    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)
    ground_plane=GroundPlane(prim_path="/World/ground_plane",size=500)

    # Choose a random spawn offset relative to the given prim
    # prim_tf = omni.usd.get_world_transform_matrix(prim)
    # spawn_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(random.uniform(-0.5, 0.5), random.uniform(3, 3.5), 0))
    # spawn_pos_gf = (spawn_offset_tf * prim_tf).ExtractTranslation()

    # Spawn pallet prim
    pallet_prim_name = "SimulatedPallet"
    pallet_prim = prims.create_prim(
        prim_path=f"{SCOPE_NAME}/{pallet_prim_name}",
        usd_path='/home/pika/Desktop/assembled/001/_converted/001_1_STL.usd',  #prefix_with_isaac_asset_server(PALLET_URL),
        semantic_label="Pallet",
    )
    print(f"{SCOPE_NAME}/{pallet_prim_name}",pallet_prim)
    # # Get the height of the pallet
    # bb_cache = create_bbox_cache()
    # curr_spawn_height = bb_cache.ComputeLocalBound(pallet_prim).GetRange().GetSize()[2] * 1.1

    # Wrap the pallet prim into a rigid prim to be able to simulate it
    pallet_rigid_prim = RigidPrim(
        prim_path=str(pallet_prim.GetPrimPath()),
        name=pallet_prim_name,
        position= Gf.Vec3d(0, 0, 10),
    )

    # Make sure physics are enabled on the rigid prim
    pallet_rigid_prim.enable_rigid_body_physics()

    # Register rigid prim with the scene
    world.scene.add(pallet_rigid_prim)

    # # Spawn boxes falling on the pallet
    # for i in range(num_boxes):
    #     # Spawn box prim
    #     cardbox_prim_name = f"SimulatedCardbox_{i}"
    #     box_prim = prims.create_prim(
    #         prim_path=f"{SCOPE_NAME}/{cardbox_prim_name}",
    #         usd_path=prefix_with_isaac_asset_server(CARDBOX_URL),
    #         semantic_label="Cardbox",
    #     )

    #     # Add the height of the box to the current spawn height
    #     curr_spawn_height += bb_cache.ComputeLocalBound(box_prim).GetRange().GetSize()[2] * 1.1

    #     # Wrap the cardbox prim into a rigid prim to be able to simulate it
    #     box_rigid_prim = RigidPrim(
    #         prim_path=str(box_prim.GetPrimPath()),
    #         name=cardbox_prim_name,
    #         position=Gf.Vec3d(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), curr_spawn_height),
    #         orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    #     )

    #     # Make sure physics are enabled on the rigid prim
    #     box_rigid_prim.enable_rigid_body_physics()

    #     # Register rigid prim with the scene
    #     world.scene.add(box_rigid_prim)

    # Reset world after adding simulated assets for physics handles to be propagated properly
    # world.reset()

    # Simulate the world for the given number of steps or until the highest box stops moving
    # last_box = world.scene.get_object(f"SimulatedCardbox_{num_boxes - 1}")
    for i in range(num_sim_steps):
        world.step(render=True)
        # if last_box and np.linalg.norm(last_box.get_linear_velocity()) < 0.001:
        #     print(f"Simulation stopped after {i} steps")
        #     break
    print('done')

    timeline=omni.timeline.get_timeline_interface()


    # timeline.set_end_time(10)
    timeline.play()
    timeline.set_start_time(0.0)
    timeline.set_end_time(10.0)

    timeline.pause()

    while True:
        simulation_app.update()

simulate_falling_objects()

