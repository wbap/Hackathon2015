from mitsuba.core import *
from mitsuba.render import *
import multiprocessing
import numpy as np
from numpy.random import *
import numpy.matlib
import json

from body import Body

from file_checker import file_checker

sensor_prop = {
    'type' : 'perspective',
    'toWorld' : Transform.lookAt(
        Point(0, 0, 95),   # Camera origin
        Point(0, 0, 0),     # Camera target
        Vector(0, 1, 0)     # 'up' vector
    ),
    'film' : {
        'type' : 'ldrfilm',
        'width' : 64,
        'height' : 64,
        'banner' : False
    },
    'sampler' : {
        'type' : 'ldsampler',
        'sampleCount' : 16
    }
}

integrator_prop = {
    'type' : 'path'
}

emitter_prop = {
        'type' : 'directional',
        'direction' :  Vector(0, 0, -1),
        'irradiance' :  Spectrum(50)
}

scheduler = Scheduler.getInstance()

# Start up the scheduling system with one worker per local core
for i in range(0, multiprocessing.cpu_count()):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

# create a queue for tracking render jobs
queue = RenderQueue()

body = Body(Point(0, 9, 0))

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

prev_left_shoulder_angles = {}

prev_right_shoulder_angles = {}

prev_left_elbow_angle = 0

prev_right_elbow_angle = 0

prev_left_hip_angles = {}

prev_right_hip_angles = {}

prev_left_knee_angle = 0

prev_right_knee_angle = 0

body.left_arm.set_joint_angles(left_shoulder_angles['p'], left_shoulder_angles['y'], left_shoulder_angles['r'], left_elbow_angle)
body.right_arm.set_joint_angles(right_shoulder_angles['p'], right_shoulder_angles['y'], right_shoulder_angles['r'], right_elbow_angle)
body.left_leg.set_joint_angles(left_hip_angles['p'], left_hip_angles['y'], left_hip_angles['r'], left_knee_angle)
body.right_leg.set_joint_angles(right_hip_angles['p'], right_hip_angles['y'], right_hip_angles['r'], right_knee_angle)

num_images = 10000

index = 0

target_list = ['mit_body_v2_%05i' % (i+1) + '.png' for i in xrange(num_images)]

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
    prev_left_hip_angles = left_hip_angles
    prev_right_hip_angles = right_hip_angles
    prev_left_knee_angle = left_knee_angle
    prev_right_knee_angle = right_knee_angle

    temp = left_shoulder_angles['p'] + randint(-1, 2)
    if (temp < shoulder_limits['p'][0]):
        left_shoulder_angles['p'] = shoulder_limits['p'][0]
    elif (temp > shoulder_limits['p'][1]):
        left_shoulder_angles['p'] = shoulder_limits['p'][1]
    else:
        left_shoulder_angles['p'] = temp

    temp = left_shoulder_angles['y'] + randint(-1, 2)
    if (temp < shoulder_limits['y'][0]):
        left_shoulder_angles['y'] = shoulder_limits['y'][0]
    elif (temp > shoulder_limits['y'][1]):
        left_shoulder_angles['y'] = shoulder_limits['y'][1]
    else:
        left_shoulder_angles['y'] = temp

    temp = left_shoulder_angles['r'] + randint(-1, 2)
    if (temp < shoulder_limits['r'][0]):
        left_shoulder_angles['r'] = shoulder_limits['r'][0]
    elif (temp > shoulder_limits['r'][1]):
        left_shoulder_angles['r'] = shoulder_limits['r'][1]
    else:
        left_shoulder_angles['r'] = temp

    temp = right_shoulder_angles['p'] + randint(-1, 2)
    if (temp < shoulder_limits['p'][0]):
        right_shoulder_angles['p'] = shoulder_limits['p'][0]
    elif (temp > shoulder_limits['p'][1]):
        right_shoulder_angles['p'] = shoulder_limits['p'][1]
    else:
        right_shoulder_angles['p'] = temp

    temp = right_shoulder_angles['y'] + randint(-1, 2)
    if (temp < shoulder_limits['y'][0]):
        right_shoulder_angles['y'] = shoulder_limits['y'][0]
    elif (temp > shoulder_limits['y'][1]):
        right_shoulder_angles['y'] = shoulder_limits['y'][1]
    else:
        right_shoulder_angles['y'] = temp

    temp = right_shoulder_angles['r'] + randint(-1, 2)
    if (temp < shoulder_limits['r'][0]):
        right_shoulder_angles['r'] = shoulder_limits['r'][0]
    elif (temp > shoulder_limits['r'][1]):
        right_shoulder_angles['r'] = shoulder_limits['r'][1]
    else:
        right_shoulder_angles['r'] = temp
            
    temp = left_elbow_angle + randint(-1, 2)
    if (temp < elbow_limits[0]):
        left_elbow_angle = elbow_limits[0]
    elif (temp > elbow_limits[1]):
        left_elbow_angle = elbow_limits[1]
    else:
        left_elbow_angle = temp

    temp = right_elbow_angle + randint(-1, 2)
    if (temp < elbow_limits[0]):
        right_elbow_angle = elbow_limits[0]
    elif (temp > elbow_limits[1]):
        right_elbow_angle = elbow_limits[1]
    else:
        right_elbow_angle = temp

    temp = left_hip_angles['p'] + randint(-1, 2)
    if (temp < hip_limits['p'][0]):
        left_hip_angles['p'] = hip_limits['p'][0]
    elif (temp > hip_limits['p'][1]):
        left_hip_angles['p'] = hip_limits['p'][1]
    else:
        left_hip_angles['p'] = temp

    temp = left_hip_angles['y'] + randint(-1, 2)
    if (temp < hip_limits['y'][0]):
        left_hip_angles['y'] = hip_limits['y'][0]
    elif (temp > hip_limits['y'][1]):
        left_hip_angles['y'] = hip_limits['y'][1]
    else:
        left_hip_angles['y'] = temp

    temp = left_hip_angles['r'] + randint(-1, 2)
    if (temp < hip_limits['r'][0]):
        left_hip_angles['r'] = hip_limits['r'][0]
    elif (temp > hip_limits['r'][1]):
        left_hip_angles['r'] = hip_limits['r'][1]
    else:
        left_hip_angles['r'] = temp

    temp = right_hip_angles['p'] + randint(-1, 2)
    if (temp < hip_limits['p'][0]):
        right_hip_angles['p'] = hip_limits['p'][0]
    elif (temp > hip_limits['p'][1]):
        right_hip_angles['p'] = hip_limits['p'][1]
    else:
        right_hip_angles['p'] = temp

    temp = right_hip_angles['y'] + randint(-1, 2)
    if (temp < hip_limits['y'][0]):
        right_hip_angles['y'] = hip_limits['y'][0]
    elif (temp > hip_limits['y'][1]):
        right_hip_angles['y'] = hip_limits['y'][1]
    else:
        right_hip_angles['y'] = temp

    temp = right_hip_angles['r'] + randint(-1, 2)
    if (temp < hip_limits['r'][0]):
        right_hip_angles['r'] = hip_limits['r'][0]
    elif (temp > hip_limits['r'][1]):
        right_hip_angles['r'] = hip_limits['r'][1]
    else:
        right_hip_angles['r'] = temp
    
    temp = left_knee_angle + randint(-1, 2)
    if (temp < knee_limits[0]):
        left_knee_angle = knee_limits[0]
    elif (temp > knee_limits[1]):
        left_knee_angle = knee_limits[1]
    else:
        left_knee_angle = temp

    temp = right_knee_angle + randint(-1, 2)
    if (temp < knee_limits[0]):
        right_knee_angle = knee_limits[0]
    elif (temp > knee_limits[1]):
        right_knee_angle = knee_limits[1]
    else:
        right_knee_angle = temp

    body.left_arm.set_joint_angles(left_shoulder_angles['p'], left_shoulder_angles['y'], left_shoulder_angles['r'], left_elbow_angle)
    body.right_arm.set_joint_angles(right_shoulder_angles['p'], right_shoulder_angles['y'], right_shoulder_angles['r'], right_elbow_angle)
    body.left_leg.set_joint_angles(left_hip_angles['p'], left_hip_angles['y'], left_hip_angles['r'], left_knee_angle)
    body.right_leg.set_joint_angles(right_hip_angles['p'], right_hip_angles['y'], right_hip_angles['r'], right_knee_angle)


    if (body.is_collision_body_parts() == False):
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
        job.start()
        # Increment index
        index += 1
    else:
        body.left_arm.set_joint_angles(prev_left_shoulder_angles['p'], prev_left_shoulder_angles['y'], prev_left_shoulder_angles['r'], prev_left_elbow_angle)
        body.right_arm.set_joint_angles(prev_right_shoulder_angles['p'], prev_right_shoulder_angles['y'], prev_right_shoulder_angles['r'], prev_right_elbow_angle)
        body.left_leg.set_joint_angles(prev_left_hip_angles['p'], prev_left_hip_angles['y'], prev_left_hip_angles['r'], prev_left_knee_angle)
        body.right_leg.set_joint_angles(prev_right_hip_angles['p'], prev_right_hip_angles['y'], prev_right_hip_angles['r'], prev_right_knee_angle)
        

    print '--------------------------------------'

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

