
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2022 Bitcraze

# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""

from controller import Robot, Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor
from controller import RangeFinder

import cv2
import numpy as np
import random
import distinctipy
import pickle
from datetime import datetime
import sys
sys.path.append('../../../../notebooks/')
from map_utils import compute_occupancy_map, get_spawn_point, get_random_rotation, axis_angle_to_euler, traj_point_gen
from label_utils import LabelGenerator
import os
import errno

from math import cos, sin

sys.path.append('../../../../controllers/')
from  pid_controller import init_pid_attitude_fixed_height_controller, pid_velocity_fixed_height_controller
from pid_controller import MotorPower_t, ActualState_t, GainsPID_t, DesiredState_t


# True to store samples
collect_data = True

# number of samples for each obstacle
reload_every = 500

# False leave the background empty, True places random obstacles
random_back = False

# ranges to sample random drone poses
yaw_l = (np.radians(-5), np.radians(5))
pitch_l = (np.radians(-5), np.radians(5))
roll_l = (np.radians(-5), np.radians(5))
z_limits = (0.45, 0.55)

# border to enlarge obstacle bounding box in occupancy map
# not used here
obstacle_margin = 0.10

# min distance from black/blue border for random pose
# not used here
border_margin = 0.70

# if False, it won't spawn the drone over a mat
# but here it doesn't really matter
spawn_on_mats = True

# webots stuff
robot = Supervisor()

# we store all the parameters and relevant experiment info in a dict
exp_dict = {"objects":{}, "env_objects":{}, "settings":{}}

# name of the experiment
if random_back:
    exp_name = "around_random_bg"
else:
    exp_name = "around_empty_bg"
# exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

exp_folder = f"../../../notebooks/full_datasets/{exp_name}"
imgs_folder = f"{exp_folder}/imgs/"
try:
    os.makedirs(imgs_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
    else:
        print(f"Overwriting folder {imgs_folder}")


exp_dict["settings"] = {
    "name": exp_name,
    "z_limits": z_limits,
    "yaw_limits": yaw_l,
    "pitch_limits": pitch_l,
    "roll_limits": roll_l,
    "spawn_on_mats": spawn_on_mats,
    "obstacle_margin": obstacle_margin,
    "border_margin": border_margin,
}

# fix seed to get always same color mapping
random.seed(42)

# here we have some convoluted webots code to store all the objects/obstacles information
objects_node = robot.getFromDef("Objects")
obj_colors = distinctipy.get_colors(objects_node.getField('children').getCount())

# we also store the obstacles explicitly to be able to move them later while setting up the scene
# i.e. to put them in the center and in the randomized background
obstacles = []
curr_obstacle = 0

for i, color in enumerate(obj_colors):
    solid = objects_node.getField('children').getMFNode(i)
    obj = {"color": solid.getField('recognitionColors').getMFColor(0)}
    obj['name'] = solid.getField('name').getSFString()

    obj['translation'] = solid.getField('translation').getSFVec3f()
    obj['rotation'] = solid.getField('rotation').getSFRotation()
    obj['scale'] = solid.getField('scale').getSFVec3f()

    if "Panel - Textured" in obj["name"] or "Mat - " in obj["name"] or "Pole - " in obj["name"]:
        # bbox info are taken from child -> $CollaAutoNames$_0 (index 0)-> child -> ID3 (index 1) -> child -> Shape at index 0?
        shape = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(0)
        pts_node = shape.getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
        pts = []
        for j in range(pts_node.getCount()):
            pts.append(pts_node.getMFVec3f(j))
        obj['points'] = pts
    elif "Panel - Metal" in obj["name"] or "Gate" in obj["name"] or "Curtain - Striped" in obj["name"]:
        # bbox is taken from all shapes in child -> $CollaAutoNames$_0 (index 0)-> child -> ID3 (index 1) -> child -> shape group -> children
        shapes = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(0).getField('children')
        pts = []
        for n in range(shapes.getCount()):
            pts_node = shapes.getMFNode(n).getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
            for j in range(pts_node.getCount()):
                pts.append(pts_node.getMFVec3f(j))
        obj['points'] = pts

    exp_dict["objects"][solid.getId()] = obj

    # this part is commented out because it can be run just once to set the objects segmentation color
    # then the scene can be saved
    # set recognition color
    # while solid.getField('recognitionColors').getCount() != 0:
    #     solid.getField('recognitionColors').removeMF(0)
    # solid.getField('recognitionColors').insertMFColor(0, list(color))
    # solid_string = solid.exportString()
    # objects_node.getField('children').removeMF(i)
    # objects_node.getField('children').importMFNodeFromString(i, solid_string)

    # we ignore the poster on the wall (we can't put it in the middle of the arena)
    if "Curtain - Striped" not in obj['name']:
        obstacles.append((solid, pts, obj['scale']))

# total number of samples
n_samples = reload_every * len(obstacles)
exp_dict['settings']['n_samples'] = n_samples


# same as above, but for environment objects (grass, walls, etc.)
objects_node = robot.getFromDef("Cyberzoo")
obj_colors = distinctipy.get_colors(objects_node.getField('children').getCount(), exclude_colors=obj_colors)

for i, color in enumerate(obj_colors):
    solid = objects_node.getField('children').getMFNode(i)
    obj = {"color": solid.getField('recognitionColors').getMFColor(0)}
    obj['name'] = solid.getField('name').getSFString()
    obj['translation'] = solid.getField('translation').getSFVec3f()
    obj['rotation'] = solid.getField('rotation').getSFRotation()
    obj['scale'] = solid.getField('scale').getSFVec3f()

    exp_dict["env_objects"][solid.getId()] = obj

    # while solid.getField('recognitionColors').getCount() != 0:
    #     solid.getField('recognitionColors').removeMF(0)
    # solid.getField('recognitionColors').insertMFColor(0, list(color))
    # solid_string = solid.exportString()
    # objects_node.getField('children').removeMF(i)
    # objects_node.getField('children').importMFNodeFromString(i, solid_string)


# this saves the scene/world. Can be left out
# robot.worldSave(robot.getWorldPath())

# this is how webots sim work. Best thing would be to read the doc.
timestep = int(robot.getBasicTimeStep())

# we keep this here only to have the rotor rotating
## Initialize motors
m1_motor = robot.getDevice("m1_motor");
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-10)
m2_motor = robot.getDevice("m2_motor");
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(10)
m3_motor = robot.getDevice("m3_motor");
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-10)
m4_motor = robot.getDevice("m4_motor");
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(10)

## Initialize Sensors
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
Keyboard().enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
camera.enableRecognitionSegmentation()
range_finder = robot.getDevice("range-finder")
range_finder.enable(timestep)
range_finder_full = robot.getDevice("range-finder-full")
range_finder_full.enable(timestep)
camera_down = robot.getDevice("camera_down")
camera_down.enable(timestep)
range_front = robot.getDevice("range_front")
range_front.enable(timestep)
range_left = robot.getDevice("range_left")
range_left.enable(timestep)
range_back = robot.getDevice("range_back")
range_back.enable(timestep)
range_right = robot.getDevice("range_right")
range_right.enable(timestep)


## Initialize struct for motor power
motorPower = MotorPower_t()

print('Take off!')
random.seed()

# store ranges for "tof" sensors
exp_dict["settings"]["range_limits"] = (range_finder.getMinRange(), range_finder.getMaxRange())
exp_dict["settings"]["range_full_limits"] = (range_finder_full.getMinRange(), range_finder_full.getMaxRange())

if collect_data:
    with open(f"{exp_folder}/exp_setup.pkl", "wb") as f:
        pickle.dump(exp_dict, f)


# Main loop:
# we need to skip the first frame because of how webots works
frame_n = -1
samples = []

while robot.step(timestep) != -1 and frame_n < n_samples:

    # first set up the scene if needed (move the next obstacle to the center and remove the previous)
    if (frame_n+1)%reload_every == 0:
        if curr_obstacle != 0: # if not the first obstacle
            # remove previous from the center
            obstacles[curr_obstacle-1][0].getField('translation').setSFVec3f([0, 10, 0])
            if random_back: # remove obstacles from the arena/random background
                for obstacle in obstacles:
                    obstacle[0].getField('translation').setSFVec3f([0, 10, 0])
        if curr_obstacle != len(obstacles):
            # move curr obstacle in the center and rotate it randomly
            obstacles[curr_obstacle][0].getField('translation').setSFVec3f([0, 0, 0.01])
            ob_rot, _ = get_random_rotation((-np.pi, np.pi), (0, 0), (0, 0))
            obstacles[curr_obstacle][0].getField("rotation").setSFRotation(ob_rot)
            # compute obstacle bounding box
            pts = np.array(obstacles[curr_obstacle][1])
            scale = obstacles[curr_obstacle][2]
            # get radius of circle around object bounding box
            x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
            y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
            w, h = np.abs(x_max - x_min)*scale[0], np.abs(y_max - y_min)*scale[1]
            min_r = np.sqrt(w**2 + h**2)/2
            max_r = min_r + 1.

            if random_back:
                # place some random objects around outside a given circle
                o_min_r = max_r
                o_max_r = o_min_r + 2.
                ids = list(range(0, len(obstacles)))
                ids.remove(curr_obstacle)
                ids = random.choices(ids, k=5)
                for idx in ids:
                    theta = np.random.uniform(0, 2*np.pi)
                    distance = np.sqrt(np.random.uniform(0, 1)*(o_max_r**2-o_min_r**2)+o_min_r**2)
                    ob_pt = [distance*np.cos(theta), distance*np.sin(theta), 0.01]
                    ob_rot, _ = get_random_rotation((-np.pi, np.pi), (0, 0), (0,0))
                    obstacles[idx][0].getField('translation').setSFVec3f(ob_pt)
                    obstacles[idx][0].getField("rotation").setSFRotation(ob_rot)

            curr_obstacle += 1


    sample = {}

    # dt = robot.getTime() - past_time;

    ## Get front camera image
    w, h = camera.getWidth(), camera.getHeight()
    cameraData = camera.getImage()  # Note: uint8 string
    # get segmentation image
    seg = camera.getRecognitionSegmentationImage()

    image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4)
    seg = np.frombuffer(seg, np.uint8).reshape(h, w, 4)

    if collect_data and frame_n>=0:
        sample["robot_position"] = spawn_pt
        sample["robot_rotation"] = random_eul_rot
        sample["sample_n"] = frame_n
        rec_objects = []
        # store info about objects recognized by the camera
        for rec_obj in camera.getRecognitionObjects():
            obj = {
            "w_id": rec_obj.get_id(),
            "camera_p": rec_obj.get_position(),
            "camera_o": rec_obj.get_orientation(),
            "camera_size": rec_obj.get_size(),
            "image_p": rec_obj.get_position_on_image(),
            "image_size": rec_obj.get_size_on_image(),
            "name": robot.getFromId(rec_obj.get_id()).getField('name').getSFString()
            }
            rec_objects.append(obj)
        sample["rec_objects"] = rec_objects
        samples.append(sample)

    ## Get down camera image
    w, h = camera_down.getWidth(), camera_down.getHeight()
    cameraData = camera_down.getImage()  # Note: uint8 string

    down_image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4)

    ## Get range images
    w, h = range_finder.getWidth(), range_finder.getHeight()
    rangeData = range_finder.getRangeImage(data_type="buffer")
    array = np.frombuffer(rangeData, np.float32)
    print("Number of elements in array:", array.size)
    range_image = np.frombuffer(rangeData, np.float32).reshape(h, w)

    w, h = range_finder_full.getWidth(), range_finder_full.getHeight()
    rangeData = range_finder_full.getRangeImage(data_type="buffer")
    range_image_full = np.frombuffer(rangeData, np.float32).reshape(h, w)

    if collect_data and frame_n>=0:
        path = f"{imgs_folder}/img_{frame_n}.png"
        cv2.imwrite(path, image)
        path = f"{imgs_folder}/down_img_{frame_n}.png"
        cv2.imwrite(path, down_image)
        path = f"{imgs_folder}/seg_{frame_n}.png"
        cv2.imwrite(path, seg)

        path = f"{imgs_folder}/range_{frame_n}.npy"
        np.save(path, range_image)

        path = f"{imgs_folder}/range_full_{frame_n}.npy"
        np.save(path, range_image_full)
    frame_n += 1

    # compute next random pose for drone
    theta = np.random.uniform(0, 2*np.pi)
    distance = np.sqrt(np.random.uniform(0, 1)*(max_r**2-min_r**2)+min_r**2)

    spawn_pt = [distance*np.cos(theta), distance*np.sin(theta), np.random.uniform(z_limits[0], z_limits[1])]
    # rotate it towards obstacle
    yaw_to_obs = np.arctan2(-spawn_pt[1], -spawn_pt[0])
    # with noise
    random_rot, random_eul_rot = get_random_rotation((yaw_to_obs+yaw_l[0], yaw_to_obs+yaw_l[1]), pitch_l, roll_l)

    # set next robot pose
    robot.getSelf().getField("translation").setSFVec3f(spawn_pt)
    robot.getSelf().getField("rotation").setSFRotation(random_rot)
    robot.getSelf().resetPhysics()

if collect_data:
    with open(f"{exp_folder}/samples.pkl", "wb") as f:
        pickle.dump(samples, f)
