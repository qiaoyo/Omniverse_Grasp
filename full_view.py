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

with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.0/Omniverse_Grasp/config.yaml') as Config_file:
    Config_yaml=yaml.load(Config_file,Loader=yaml.FullLoader)

kit=SimulationApp(launch_config=Config_yaml['WorldConfig'])

data_path='/home/pika/After_Falling.usd'
data_path='/home/pika/assemble_step/078/_converted/078_1.usd'
prim_path="/World/Stand"

from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
my_world=World(stage_units_in_meters=0.01)
# create_prim(
#             prim_path=prim_path,
#             # position=position,
#             # orientation=orientation,
#             # scale=[part_scale,part_scale,part_scale],
#             usd_path=data_path,
#             # semantic_label=semantic_label
#         )
stand = add_reference_to_stage(usd_path=data_path, prim_path="/World/Stand")
while True:
    kit.update()
