from mitsuba.core import *
from mitsuba.render import *
import multiprocessing
import numpy as np
from numpy.random import *
import numpy.matlib
import json

from body import Body

from file_checker import file_checker

target_list = ['mit_body_v2_00004.json']

sensor_prop = {
        'type' : 'perspective',
        'fov': 120.0,
        'toWorld' : Transform.lookAt(
            Point(0, 15+9, 25),   # Camera origin
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
        'direction' :  Vector(0, -1, 0),
        'irradiance' :  Spectrum(50)
}

body = Body(Point(0, 9, 30))
    
body.left_arm.set_slocal_rotate_angle(0, 0, 180)
body.right_arm.set_slocal_rotate_angle(0, 0, 180)
body.left_leg.set_hlocal_rotate_angle(0, 0, 180)
body.right_leg.set_hlocal_rotate_angle(0, 0, 180)

# create a queue for tracking render jobs
queue = RenderQueue()

# rendering again
for target in target_list:
    # open json file
    with open(target) as f:
        json_data = json.load(f)
    # Set angle parameters from json data (not json file!)
    body.set_from_json_data(json_data)
    destination = target.split('.')[0]
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
    job = RenderJob('myRenderJob' + target, scene, queue)
    job.start()

# wait for all jobs to finish and release resource
queue.waitLeft(0)
queue.join()

# Print some statistics about the rendering process
print(Statistics.getInstance().getStats())
