from mitsuba.core import *
from mitsuba.render import *
import multiprocessing
import numpy as np
from numpy.random import *
import numpy.matlib
import json
import time
from PIL import Image

from body import Body

from file_checker import file_checker

# settings
lower_body_flag = True
first_person_view_flag = True
lr_flip_flag = False 
generate_json_flag = False

sensor_prop = {'type' : 'perspective'}

if first_person_view_flag:
    sensor_prop['fov'] = 120.0
    
    sensor_prop['toWorld'] = Transform.lookAt(
        Point(0, 15+9, 3.5),   # Camera origin
        Point(0, 9+9, 28.5),     # Camera target
        Vector(0, 1, 0)     # 'up' vector
    )
else:
    sensor_prop['toWorld'] = Transform.lookAt(
        Point(0, 0, 95),   # Camera origin
        Point(0, 0, 0),     # Camera target
        Vector(0, 1, 0)     # 'up' vector
    )

sensor_prop['film'] = {
    'type' : 'ldrfilm',
    'width' : 64,
    'height' : 64,
    'banner' : False
}

sensor_prop['sample'] = {
    'type' : 'ldsampler',
    'sampleCount' : 16
}


integrator_prop = {
    'type' : 'path'
}


emitter_prop = {'type' : 'directional'}
emitter_prop['irradiance'] = Spectrum(50)

if first_person_view_flag:
    emitter_prop['direction'] = Vector(0, -1, 0)
else:
    emitter_prop['direction'] = Vector(0, 0, -1)

scheduler = Scheduler.getInstance()

# Start up the scheduling system with one worker per local core
for i in range(0, multiprocessing.cpu_count()):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

# create a queue for tracking render jobs
queue = RenderQueue()

body = Body(Point(0, 9, 0), lower_body_flag)

shoulder_limits = {
        'p': [-180, 25],
        'y': [-90, 30],
        'r': [-180 , 180]
}

elbow_limits = [0, 135]

hip_limits = {
        'p': [-125, 15],
        'y': [-90, 60],
        'r': [-30, 70]
}

knee_limits = [0, 150]

left_shoulder_angles = {
        'p': -85,
        'y': -30,
        'r': 0
}

right_shoulder_angles = {
        'p': -85,
        'y': -30,
        'r': 0 
}

left_elbow_angle = 70

right_elbow_angle = 70

prev_left_shoulder_angles = {}

prev_right_shoulder_angles = {}

prev_left_elbow_angle = 0

prev_right_elbow_angle = 0

body.left_arm.set_joint_angles(left_shoulder_angles['p'], left_shoulder_angles['y'], left_shoulder_angles['r'], left_elbow_angle)
body.right_arm.set_joint_angles(right_shoulder_angles['p'], right_shoulder_angles['y'], right_shoulder_angles['r'], right_elbow_angle)

if lower_body_flag:
    left_hip_angles = {
            'p': -55,
            'y': -15,
            'r': 20 
    }
    
    right_hip_angles = {
            'p': -55,
            'y': -15,
            'r': 20
    }
    
    left_knee_angle = 75
    
    right_knee_angle = 75
    
    prev_left_hip_angles = {}
    
    prev_right_hip_angles = {}
    
    prev_left_knee_angle = 0
    
    prev_right_knee_angle = 0
    
    body.left_leg.set_joint_angles(left_hip_angles['p'], left_hip_angles['y'], left_hip_angles['r'], left_knee_angle)
    body.right_leg.set_joint_angles(right_hip_angles['p'], right_hip_angles['y'], right_hip_angles['r'], right_knee_angle)

num_images = 100 

index = 0

target_list = ['mit_body_v2_%05i' % (i+1) + '.png' for i in xrange(num_images)]
if not generate_json_flag:
    json_list = ['mit_body_v2_%05i' % (i+1) + '.json' for i in xrange(num_images)]

if generate_json_flag:
    # Normal run (generating json & png images)
    while (index < num_images):
        destination = 'mit_body_v2_%05i' % (index+1)
        scene = Scene()
        pmgr = PluginManager.getInstance()
        # Create a sensor, film & sample generator
        scene.addChild(pmgr.create(sensor_prop))
        # Set the integrator
        scene.addChild(pmgr.create(integrator_prop))
        # Add a light source
        scene.addChild(pmgr.create(emitter_prop))
    
        prev_left_shoulder_angles = left_shoulder_angles
        prev_right_shoulder_angles = right_shoulder_angles
        prev_left_elbow_angle = left_elbow_angle
        prev_right_elbow_angle = right_elbow_angle
    
        if lower_body_flag:
            prev_left_hip_angles = left_hip_angles
            prev_right_hip_angles = right_hip_angles
            prev_left_knee_angle = left_knee_angle
            prev_right_knee_angle = right_knee_angle
    
        temp1 = left_shoulder_angles['p'] + randint(-1, 2)
        if ((temp1 < shoulder_limits['p'][0]) or (temp1 > shoulder_limits['p'][1])):
            continue
    
        temp2 = left_shoulder_angles['y'] + randint(-1, 2)
        if ((temp2 < shoulder_limits['y'][0]) or (temp2 > shoulder_limits['y'][1])):
            continue
    
        temp3 = left_shoulder_angles['r'] + randint(-1, 2)
        if ((temp3 < shoulder_limits['r'][0]) or (temp3 > shoulder_limits['r'][1])):
            continue
    
        temp4 = right_shoulder_angles['p'] + randint(-1, 2)
        if ((temp4 < shoulder_limits['p'][0]) or (temp4 > shoulder_limits['p'][1])):
            continue
    
        temp5 = right_shoulder_angles['y'] + randint(-1, 2)
        if ((temp5 < shoulder_limits['y'][0]) or (temp5 > shoulder_limits['y'][1])):
            continue
    
        temp6 = right_shoulder_angles['r'] + randint(-1, 2)
        if ((temp6 < shoulder_limits['r'][0]) or (temp6 > shoulder_limits['r'][1])):
            continue
                
        temp7 = left_elbow_angle + randint(-1, 2)
        if ((temp7 < elbow_limits[0]) or (temp7 > elbow_limits[1])):
            continue
    
        temp8 = right_elbow_angle + randint(-1, 2)
        if ((temp8 < elbow_limits[0]) or (temp8 > elbow_limits[1])):
            continue
        
        if lower_body_flag:
            temp9 = left_hip_angles['p'] + randint(-1, 2)
            if ((temp9 < hip_limits['p'][0]) or (temp9 > hip_limits['p'][1])):
                continue
    
            temp10 = left_hip_angles['y'] + randint(-1, 2)
            if ((temp10 < hip_limits['y'][0]) or (temp10 > hip_limits['y'][1])):
                continue
    
            temp11 = left_hip_angles['r'] + randint(-1, 2)
            if ((temp11 < hip_limits['r'][0]) or (temp11 > hip_limits['r'][1])): 
                continue
    
            temp12 = right_hip_angles['p'] + randint(-1, 2)
            if ((temp12 < hip_limits['p'][0]) or (temp12 > hip_limits['p'][1])):
                continue
    
            temp13 = right_hip_angles['y'] + randint(-1, 2)
            if ((temp13 < hip_limits['y'][0]) or (temp13 > hip_limits['y'][1])):
                continue
    
            temp14 = right_hip_angles['r'] + randint(-1, 2)
            if ((temp14 < hip_limits['r'][0]) or (temp14 > hip_limits['r'][1])):
                continue
            
            temp15 = left_knee_angle + randint(-1, 2)
            if ((temp15 < knee_limits[0]) or (temp15 > knee_limits[1])):
                continue
    
            temp16 = right_knee_angle + randint(-1, 2)
            if ((temp16 < knee_limits[0]) or (temp16 > knee_limits[1])):
                continue
    
        left_shoulder_angles = dict(zip(['p', 'y', 'r'], [temp1, temp2, temp3]))
        right_shoulder_angles = dict(zip(['p', 'y', 'r'], [temp4, temp5, temp6]))
        left_elbow_angle = temp7
        right_elbow_angle = temp8
        
        if lower_body_flag:
            left_hip_angles = dict(zip(['p', 'y', 'r'], [temp9, temp10, temp11]))
            right_hip_angles = dict(zip(['p', 'y', 'r'], [temp12, temp13, temp14]))
            left_knee_angle = temp15
            right_knee_angle = temp16
        
        body.left_arm.set_joint_angles(left_shoulder_angles['p'], left_shoulder_angles['y'], left_shoulder_angles['r'], left_elbow_angle)
        body.right_arm.set_joint_angles(right_shoulder_angles['p'], right_shoulder_angles['y'], right_shoulder_angles['r'], right_elbow_angle)
    
        if lower_body_flag:
            body.left_leg.set_joint_angles(left_hip_angles['p'], left_hip_angles['y'], left_hip_angles['r'], left_knee_angle)
            body.right_leg.set_joint_angles(right_hip_angles['p'], right_hip_angles['y'], right_hip_angles['r'], right_knee_angle)
    
        cur_left_elbow = body.left_arm.calc_elbow_point()
        cur_right_elbow = body.right_arm.calc_elbow_point()
        cur_left_hand = body.left_arm.calc_hand_point()
        cur_left_hand = body.right_arm.calc_hand_point()
    
        departure_flag = (cur_left_elbow.z < 0) or (cur_right_elbow.z < 0) or (cur_left_hand.z < 0) or (cur_left_hand.z < 0)
    
        if ((body.is_collision_body_parts() == False) and (departure_flag == False)):
            print '\033[94mcollision detection passed!\033[0m'
            # Add body parts
            for prop in body.get_property_list():
                scene.addChild(pmgr.create(prop))
    
            json_data = body.constr_json_data()
                    
            with open('mit_body_v2_%05i.json' % (index + 1), 'w') as f:
                json.dump(json_data, f, indent=2)
            
            scene.configure()
            scene.setDestinationFile(destination)
    
            # Create a render job and insert it into the queue
            job = RenderJob('myRenderJob' + str(index), scene, queue)
    
            while True:
                # Check current number of thread
                num_thread = queue.getJobCount()
                if num_thread > 1000:
                    time.sleep(1)
                else:
                    break
    
            job.start()
            # Increment index
            index += 1
        else:
            body.left_arm.set_joint_angles(prev_left_shoulder_angles['p'], prev_left_shoulder_angles['y'], prev_left_shoulder_angles['r'], prev_left_elbow_angle)
            body.right_arm.set_joint_angles(prev_right_shoulder_angles['p'], prev_right_shoulder_angles['y'], prev_right_shoulder_angles['r'], prev_right_elbow_angle)
            left_shoulder_angles['p'] = prev_left_shoulder_angles['p']
            left_shoulder_angles['y'] = prev_left_shoulder_angles['y']
            left_shoulder_angles['r'] = prev_left_shoulder_angles['r']
            right_shoulder_angles['p'] = prev_right_shoulder_angles['p']
            right_shoulder_angles['y'] = prev_right_shoulder_angles['y']
            right_shoulder_angles['r'] = prev_right_shoulder_angles['r']
            left_elbow_angle = prev_left_elbow_angle
            right_elbow_angle = prev_right_elbow_angle
    
            if lower_body_flag:
                body.left_leg.set_joint_angles(prev_left_hip_angles['p'], prev_left_hip_angles['y'], prev_left_hip_angles['r'], prev_left_knee_angle)
                body.right_leg.set_joint_angles(prev_right_hip_angles['p'], prev_right_hip_angles['y'], prev_right_hip_angles['r'], prev_right_knee_angle)
                left_hip_angles['p'] = prev_left_hip_angles['p']
                left_hip_angles['y'] = prev_left_hip_angles['y']
                left_hip_angles['r'] = prev_left_hip_angles['r']
                right_hip_angles['p'] = prev_right_hip_angles['p']
                right_hip_angles['y'] = prev_right_hip_angles['y']
                right_hip_angles['r'] = prev_right_hip_angles['r']
                left_knee_angle = prev_left_knee_angle
                right_knee_angle = prev_right_knee_angle
    
        print '--------------------------------------'
    
    # wait for all jobs to finish and release resource
    queue.waitLeft(0)
    queue.join()
    
    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())
else:
    # re-rendering run (generating only png images)
    for index, target_json in enumerate(json_list):
        # open json file
        with open(target_json) as f:
            json_data = json.load(f)
        # Set angle parameters from json data (not json file!)
        body.set_from_json_data(json_data)
        destination = target_json.split('.')[0]
        scene = Scene()
        pmgr = PluginManager.getInstance()
        # Create a sensor, film & sample generator
        scene.addChild(pmgr.create(sensor_prop))
        # Set the integrator
        scene.addChild(pmgr.create(integrator_prop))
        # Add a light source
        scene.addChild(pmgr.create(emitter_prop))
        # Add body parts
        for prop in body.get_property_list():
            scene.addChild(pmgr.create(prop))
    
        scene.configure()
        scene.setDestinationFile(destination)
    
        # Create a render job and insert it into the queue
        job = RenderJob('myRenderJob' + str(index), scene, queue)
        job.start()
    
    # wait for all jobs to finish and release resource
    queue.waitLeft(0)
    queue.join()
    
    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())

while True:
    print 'Now, Check rendering results and repeat rendering'
    # check png file existence
    flag_list = file_checker(target_list)
    if flag_list.count(False) == 0:
        break

    # create a queue for tracking render jobs
    queue = RenderQueue()

    # rendering again
    for i, flag in enumerate(flag_list):
        if not flag:
            # open json file
            with open('mit_body_v2_%05i.json' % (i+1)) as f:
                json_data = json.load(f)
            # Set angle parameters from json data (not json file!)
            body.set_from_json_data(json_data)
            destination = 'mit_body_v2_%05i' % (i+1)

            scene = Scene()
            pmgr = PluginManager.getInstance()
            # Create a sensor, film & sample generator
            scene.addChild(pmgr.create(sensor_prop))
            # Set the integrator
            scene.addChild(pmgr.create(integrator_prop))
            # Add a light source
            scene.addChild(pmgr.create(emitter_prop))
            # Add body parts
            for prop in body.get_property_list():
                scene.addChild(pmgr.create(prop))

            scene.configure()
            scene.setDestinationFile(destination)

            # Create a render job and insert it into the queue
            job = RenderJob('myRenderJob' + str(i), scene, queue)
            job.start()

    # wait for all jobs to finish and release resource
    queue.waitLeft(0)
    queue.join()

    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())

# left right flip process (Third-persoon point of view vs. Mirror image)
if lr_flip_flag:
    print 'Now, generating mirror images'
    for target in target_list:
        img = Image.open(target)
        lr_flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        lr_flip_img.save(target.split('.')[0] + 'mirror.png')
