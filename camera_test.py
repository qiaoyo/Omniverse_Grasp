import os,sys
import numpy as np
import yaml
import time


def main():

    import omni
    from omni.isaac.kit import SimulationApp
    CONFIG = {"width": 1920, "height": 1080, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

    # with open('/home/pika/.local/share/ov/pkg/isaac_sim-2022.2.1/assemble/config.yaml') as f:
    #     config=yaml.load(f)
    
    simulation_app=SimulationApp(launch_config=CONFIG)

    from omni.isaac.core.utils.stage import get_current_stage,get_stage_units
    from omni.isaac.core import World
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.materials import PhysicsMaterial,PreviewSurface
    import omni.usd
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    # from omni.physx.scripts.physicsUtils import *
    from pxr import Gf,Sdf,UsdPhysics,PhysicsSchemaTools,UsdShade,UsdGeom,UsdLux,Tf,Vt,PhysxSchema
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from contactimmediate import ContactReportDirectAPIDemo
    import omni.kit.commands
    from omni.isaac.core.utils import prims
    import omni.replicator.core as rep

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
        scale=[1,1,1],
        usd_path=usd_path,
        semantic_label='056_1'
    )

    component_prim_1=stage.GetPrimAtPath("/World/Assemble_1")
    utils.setRigidBody(component_prim_1,"convexHull",False)



    writer=rep.WriterRegistry.get("BasicWriter")
    output_directory = '/home/pika/Desktop/assembled/_output_headless'
    print("Outputting data to ", output_directory)
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        # bounding_box_2d_tight=True,
        # semantic_segmentation=True,
        # instance_segmentation=True,
        # distance_to_image_plane=True,
        # distance_to_camera=True,
        # bounding_box_3d=True,
        # occlusion=True,
        # normals=True,
    )

    rep_product_list=[]
    for i in range(5):

        rep_camera=rep.create.camera(
        focus_distance=400.0, 
        focal_length=24.0, 
        clipping_range=(0.1, 10000000.0), 
        name="Camera"
        )
    

        with rep_camera:
            rep.modify.pose(
                position=(100,50,300),
                rotation=(0,-90,0)
            )



        RESOLUTION=(CONFIG['width'],CONFIG['height'])
        rep_product=rep.create.render_product(rep_camera,RESOLUTION)
        rep_product_list.append(rep_product)

    writer.attach(rep_product_list)
    # render_product=rep.create.render_product(rep_camera,resolution=(640,480))

    
    # # rgb_data=rep.AnnotatorRegistry.get_annotator("rgb")
    # # depth_data=rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    # # rgb_data.attach([render_product])
    # # depth_data.attach([render_product])

    for i in range(5):
        rep.orchestrator.step()

    # while True:
    #     simulation_app.update()

    
    simulation_app.close()


def get_point_cloud_from_z(Y,camera_matrix,scale=1):

    x,z=np.meshgrid(np.arange(Y.shape[-1]),
                    np.arange(Y.shape[-2]-1,-1,-1))
    for i in range(Y.ndim-2):
        x=np.expand_dims(x,axis=0)
        z=np.expand_dims(z,axis=0)

    X=(x[::scale,::scale]-camera_matrix.xc)*Y[::scale,::scale]/camera_matrix.f
    Z=(z[::scale,::scale]-camera_matrix.zc)*Y[::scale,::scale]/camera_matrix.f

    XYZ=np.concatenate((X[...,np.newaxis],Y[::scale,::scale][...,np.newaxis],Z[...,np.newaxis]),axis=X.ndim)

    return XYZ

def test_rgbd():
    import cv2
    from PIL import Image
    depth_path='/home/pika/Desktop/assembled/_output_headless/distance_to_camera_0000.npy'
    rgb_path='~/Desktop/assembled/_output_headless/rgb_0000.png'

    im_depth=np.load(depth_path)
    # im_depth=Image.fromarray(im_depth)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
    img=Image.fromarray(im_color)
    
    from argparse import Namespace
    from mpl_toolkits.mplot3d import Axes3D
    camera_matrix={'zc':959.5,'xc':539.5,'f':24}
    camera_matrix=Namespace(**camera_matrix)

    XYZ=get_point_cloud_from_z(im_depth,camera_matrix)
    print(XYZ.shape)
    import matplotlib.pyplot as plt
    import open3d as o3d
    ax = plt.figure(1).gca(projection='3d')
    ax.plot(np.ndarray.flatten(XYZ[::,::,0]),np.ndarray.flatten(XYZ[::,::,1]),np.ndarray.flatten(XYZ[::,::,2]),'b.',markersize=0.2)
    plt.title('point cloud')
    plt.show()

    
    XYZ=XYZ.reshape(-1,3)
    vis=o3d.visualization.Visualizer()
    vis.create_window(window_name='pc')
    vis.get_render_option().point_size=1
    opt=vis.get_render_option()
    opt.background_color=np.asarray([0,0,0])

    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points=o3d.open3d.utility.Vector3dVector(XYZ)
    pcd.paint_uniform_color([1,1,1])
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print('finish!')


