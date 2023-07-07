import omni.replicator.core as rep

camera1_pos = [(0, 100, 500),(500, 100, 500),(-500, 100, 500),]
camera2_pos = [(0, 500, 0),(500, 500, 0),(-500, 500, 0),]

with rep.new_layer():
    cube = rep.create.cube(position=(200, 0, 0), semantics=[('class', 'cube')])
    sphere = rep.create.sphere(position=(0, 0, 0), semantics=[('class', 'sphere')])
    cone = rep.create.cone(position=(-200, 0, 0), semantics=[('class', 'cone')])

    camera = rep.create.camera(position=camera1_pos[0], look_at=(0,0,0))
    camera2 = rep.create.camera(position=camera2_pos[0], look_at=(0,0,0))

    with rep.trigger.on_frame(num_frames=3):
        with camera:
            rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera1_pos))
        with camera2:
            rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera2_pos))

# Will render 512x512 images and 320x240 images
render_product = rep.create.render_product(camera, (512, 512))
render_product2 = rep.create.render_product(camera2, (320, 240))

basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir=f"~/replicator_examples/multi_render_product/basic",
    rgb=True,
    bounding_box_2d_loose=True,
    bounding_box_2d_tight=True,
    bounding_box_3d=True,
    distance_to_camera=True,
    distance_to_image_plane=True,
    instance_segmentation=True,
    normals=True,
    semantic_segmentation=True,
)
# Attach render_product to the writer
basic_writer.attach([render_product, render_product2])
# Run the simulation graph
rep.orchestrator.run()
