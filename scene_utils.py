import cv2
import numpy as np
import os
import math
from scipy.spatial import cKDTree
import json
import genesis as gs
import sys


def line_segment_aabb_intersect_2d(old_pos, new_pos, aabb_min, aabb_max):
    """
    2D line segment (old_pos -> new_pos) vs 2D AABB (min, max) intersection test.
    AABB is given as (xmin, ymin), (xmax, ymax).

    Returns:
        (ok, t_enter):
          ok=True  -> intersection exists for 0 <= t_enter <= 1
          ok=False -> no intersection
    """
    epsilon = 1e-9
    dxdy = new_pos - old_pos

    length_sq = dxdy[0] * dxdy[0] + dxdy[1] * dxdy[1]
    if length_sq < epsilon:
        # Degenerate case: effectively a point; one could check if it lies inside AABB.
        # For simplicity, we treat this as "no collision" here.
        return (False, None)
    # if np.allclose(dxdy, 0):
    #     # Segment length 0 => cannot compute a valid intersection
    #     return (False, None)

    inv = np.zeros(2, dtype=np.float32)
    for i in range(2):
        if abs(dxdy[i]) < epsilon:
            inv[i] = 1e12  # pseudo-infinity
        else:
            inv[i] = 1.0 / dxdy[i]

    tmin = -1e12
    tmax = 1e12
    O = old_pos
    D = dxdy

    # x, y axes (2D) slabs
    # axis=0 -> x, axis=1 -> y
    for axis in range(2):
        slab_min = aabb_min[axis]  # xmin or ymin
        slab_max = aabb_max[axis]  # xmax or ymax

        t1 = (slab_min - O[axis]) * inv[axis]
        t2 = (slab_max - O[axis]) * inv[axis]

        t_enter_axis = min(t1, t2)
        t_exit_axis = max(t1, t2)

        if t_enter_axis > tmin:
            tmin = t_enter_axis
        if t_exit_axis < tmax:
            tmax = t_exit_axis
        if tmax < tmin:
            return (False, None)

    # Restrict to segment: t in [0, 1]
    if tmax < 0 or tmin > 1:
        return (False, None)
    t_enter = max(0.0, tmin)
    return (True, t_enter)


def line_segment_colliding_with_obstacles_2d_obb(old_pos, new_pos, obstacles_2d):
    """
    Check collision between a 2D line segment and a list of 2D OBB obstacles.

    Args:
        obstacles_2d: list of tuples (center, angle_deg, half_w, half_h)
        old_pos, new_pos: (x, y) endpoints of the segment

    Returns:
        (collided, t_min):
          collided=True  -> smallest intersection t_enter
          collided=False -> no collision, returns 1.0
    """
    collided = False    # global flag
    t_min_global = 1.0  # smallest t_enter among all obstacles

    for (center, angle_deg, half_w, half_h) in obstacles_2d:
        ok, t_enter = line_segment_vs_obb_2d(
            old_pos, new_pos, center, angle_deg, half_w, half_h
        )
        if ok:
            # If intersection occurs, keep the smallest t_enter
            if t_enter < t_min_global:
                t_min_global = t_enter
                collided = True

    if collided:
        return True, t_min_global
    else:
        return False, 1.0


def line_segment_vs_obb_2d(old_pos, new_pos, center, angle_deg, half_w, half_h, epsilon=1e-9):
    """
    2D line segment vs rotated rectangle (OBB) intersection test.

    Args:
        center: (2,) OBB center
        angle_deg: float, rotation angle in degrees (around the x-axis in 2D sense)
        half_w, half_h: float, half width and half height of the OBB

    Returns:
        (ok, t_enter):
          ok=True  -> intersection exists for 0 <= t_enter <= 1
          ok=False -> no intersection
    """
    # 1) Transform the segment into OBB local coordinates
    #    angle_deg may need clockwise/counterclockwise adjustment. For example,
    #    OpenCV's minAreaRect often returns angle<0 meaning CCW, so rotating by
    #    -angle can align the OBB with axes.
    theta_rad = math.radians(-angle_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    def transform_to_local(pt):
        # translate
        px = pt[0] - center[0]
        py = pt[1] - center[1]
        # rotate by -angle
        lx = px * cos_t - py * sin_t
        ly = px * sin_t + py * cos_t
        return np.array([lx, ly], dtype=np.float32)

    old_local = transform_to_local(old_pos)
    new_local = transform_to_local(new_pos)

    # 2) In local space, OBB becomes an AABB: ([-half_w, half_w], [-half_h, half_h])
    aabb_min = np.array([-half_w, -half_h], dtype=np.float32)
    aabb_max = np.array([half_w, half_h], dtype=np.float32)

    return line_segment_aabb_intersect_2d(old_local, new_local, aabb_min, aabb_max)


def load_obstacles_2d(filename='obstacles_2d.json'):
    import json, numpy as np
    with open(filename, 'r') as f:
        raw_list = json.load(f)
    # raw_list[i] = [cx, cy, angle_deg, half_w, half_h]
    obstacles = []
    for rec in raw_list:
        cx, cy, angle_deg, half_w, half_h = rec
        center_np = np.array([cx, cy], dtype=np.float32)
        obstacles.append((center_np, angle_deg, half_w, half_h))
    print(f"Loaded {len(obstacles)} obstacles from {filename}")
    return obstacles


# for training

def bring_scene(scene_num, scene, scene_data_path, occ_vis=True):
    dict_entities = {}
    with open(scene_data_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    scene_key = f"scene{scene_num:02d}"

    if scene_key not in scene_data:
        print(f"Scene {scene_num} not found in {scene_data_path}")
        return dict_entities

    entity_list = scene_data[scene_key]

    for entity in entity_list:
        name = entity["name"]
        file_path = os.path.join('/external_asset/_occluders', entity["file_path"])
        pos = tuple(entity["pos"])
        euler = tuple(entity["euler"])
        scale = entity["scale"]

        new_entity = scene.add_entity(
            gs.morphs.Mesh(
                file=file_path,
                pos=pos,
                euler=euler,
                scale=scale,
                fixed=True,
                visualization=occ_vis
            )
        )
        dict_entities[name] = new_entity

    return dict_entities


def bring_house(house_num, datapath='/_targets/_50_targets.json'):
    print("THIS IS THE HOUSE-3K HOUSE")
    with open(datapath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Example: if item["name"] == "B_14", then split('_') → ["B", "14"]
        splitted = item["name"].split('_')
        if len(splitted) > 1:
            # Convert the trailing number to int and compare
            num = int(splitted[-1])
            if num == house_num:
                return item["file_path"], item["name"], item["max_cr"]
    raise ValueError(f"cant find house_num={house_num}in data")


def bring_house_omniobject3d(house_num, datapath='/_targets/_OmniObject3D_house_targets.json'):
    print("THIS IS THE OMNIOBJECT3D HOUSE")
    with open(datapath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Example: if item["name"] == "B_14", then split('_') → ["B", "14"]
        splitted = item["name"].split('_')
        if len(splitted) > 1:
            # Convert the trailing number to int and compare
            num = int(splitted[-1])
            if num == house_num:
                return (
                    item["file_path"],
                    item["name"],
                    item["max_cr"],
                    item['height'],
                    item['scale']
                )
    raise ValueError(f"cant find house_num={house_num}in data")


def train_scene_build(scene_num, house_num, show_viewer=False, render=False, ray=False):
    try:
        gs.init(backend=gs.gpu)
    except:
        gs.destroy()
        gs.init(backend=gs.gpu)

    target_path, target_name, max_cr = bring_house(house_num)
    print(f"scene {scene_num} house {house_num} target {target_name} started!!!!")

    if not ray:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            # renderer=gs.renderers.RayTracer(),
        )
    else:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="/map/sky.png",
                    ),
                ),
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (5.0, 10.0, 20.0), "radius": 3.5,
                     "color": (60.0, 60.0, 30.0), "intesity": 10.0},
                ],
            ),
        )
    # keep the entity order especially for the
    # 0) house, 1) plane, 2)... occluders, -1) robot
    target_object = scene.add_entity(
        gs.morphs.Mesh(
            file=f'/_targets/{target_path}',
            pos=(0, 0, 0),
            euler=(90, 0, 0),
            scale=15,
            # collision=True,
            # convexify=False,
            fixed=True
        )
    )

    # plane = scene.add_entity(gs.morphs.Plane())
    plane = scene.add_entity(
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="/external_asset/house/grass.jpg",
            )
        ),
    )

    ##########################################
    # 2) Get configs
    ##########################################

    ##########################################
    # 3) Setup the scene
    ##########################################
    scene_data_path = "/_targets/_scene_data.json"
    entities_dict = bring_scene(scene_num, scene, scene_data_path)
    entities = [target_object] + list(entities_dict.values())

    ##########################################
    # 3) Set camera
    ##########################################
    cam = scene.add_camera(
        # res=(640, 480)
        res=(320, 240),
        # res=(1280, 720),
        fov=60,
        # GUI=True,
        GUI=render,
        pos=(20, 20, 15),
        lookat=(0, 0, 0.4),
    )
    ##########################################
    # 4) Scene Build & Env Setup
    ##########################################
    scene.build(n_envs=1)
    return scene, cam, entities, max_cr


def test_scene_build_omniobject3d(scene_num, house_num, show_viewer=False, render=False, ray=False):
    try:
        gs.init(backend=gs.gpu)
    except:
        gs.destroy()
        gs.init(backend=gs.gpu)

    if not ray:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            # renderer=gs.renderers.RayTracer(),
        )
    else:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="/map/sky.png",
                    ),
                ),
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (5.0, 10.0, 20.0), "radius": 3.5,
                     "color": (60.0, 60.0, 30.0), "intesity": 10.0},
                ],
            ),
        )
    # ################# omniobject3d ##############################################################

    target_path, target_name, max_cr, height, scale = bring_house_omniobject3d(house_num)
    # keep the entity order especially for the
    # 0) house, 1) plane, 2)... occluders, -1) robot
    print(f"scene {scene_num} house {house_num} target {target_name} started!!!!")
    print(f"target_path: {target_path}")
    print(f"height: {height}")
    print(f"scale: {scale}")

    target_object = scene.add_entity(
        gs.morphs.Mesh(
            file=f'/_targets/{target_path}',
            pos=(0, 0, height),
            euler=(90, 0, 0),
            scale=scale,
            # collision=True,
            # convexify=False,
        )
    )

    ##########################################
    # 2) Add scenes
    ##########################################

    # plane = scene.add_entity(gs.morphs.Plane())
    plane = scene.add_entity(
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="/external_asset/house/grass.jpg",
            )
        ),
    )

    ##########################################
    # 3) Setup the scene
    ##########################################
    scene_data_path = "/_targets/_scene_data.json"
    entities_dict = bring_scene(scene_num, scene, scene_data_path)
    entities = [target_object] + list(entities_dict.values())

    ##########################################
    # 3) Set camera
    ##########################################
    cam = scene.add_camera(
        # res=(640, 480),
        res=(320, 240),
        fov=60,
        GUI=render,
        pos=(20, 20, 15),
        lookat=(0, 0, 0.4),
    )
    ##########################################
    # 4) Scene Build & Env Setup
    ##########################################
    scene.build(n_envs=1)

    return scene, cam, entities, max_cr


def test_scene_build_house3k(scene_num, house_num, show_viewer=False, render=False, ray=False):
    try:
        gs.init(backend=gs.gpu)
    except:
        gs.destroy()
        gs.init(backend=gs.gpu)

    if not ray:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            # renderer=gs.renderers.RayTracer(),
        )
    else:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=int(0.5 / 0.02)
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False, n_rendered_envs=1),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="./map/sky.png",
                    ),
                ),
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (5.0, 10.0, 20.0), "radius": 3.5,
                     "color": (60.0, 60.0, 30.0), "intesity": 10.0},
                ],
            ),
        )

    #################### house3k-occ ###########################################################
    target_path, target_name, max_cr = bring_house(house_num)
    # overwrite the target to test set E
    target_name = "E" + target_name[1:]
    # Extract the letter from the path pattern
    import re
    match = re.search(r"Set_([A-Z])\/BAT1_SET\1_", target_path)
    if match:
        new_char = "E"
        old_char = match.group(1)
        # Replace the letter in the path
        target_path = target_path.replace(f"Set_{old_char}", f"Set_{new_char}")
        target_path = target_path.replace(f"SET{old_char}", f"SET{new_char}")
    print(f"scene {scene_num} house {house_num} target {target_name} started!!!!")
    print(f"target_path: {target_path}")
    # keep the entity order especially for the
    # 0) house, 1) plane, 2)... occluders, -1) robot
    target_object = scene.add_entity(
        gs.morphs.Mesh(
            file=f'/_targets/{target_path}',
            pos=(0, 0, 0),
            euler=(90, 0, 0),
            scale=15,
            # collision=True,
            # convexify=False,
        )
    )
    ##############################################################################################

    ##########################################
    # 2) Add scenes
    ##########################################

    # plane = scene.add_entity(gs.morphs.Plane())
    plane = scene.add_entity(
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="/external_asset/house/grass.jpg",
            )
        ),
    )

    ##########################################
    # 3) Setup the scene
    ##########################################
    scene_data_path = "/_targets/_scene_data.json"
    entities_dict = bring_scene(scene_num, scene, scene_data_path)
    entities = [target_object] + list(entities_dict.values())

    ##########################################
    # 3) Set camera
    ##########################################
    cam = scene.add_camera(
        # res=(640, 480),
        res=(320, 240),
        # res=(1280, 720),
        fov=60,
        GUI=render,
        pos=(20, 20, 15),
        lookat=(0, 0, 0.4),
    )
    ##########################################
    # 4) Scene Build & Env Setup
    ##########################################
    scene.build(n_envs=1)
    return scene, cam, entities, max_cr
