from mitsuba.core import *
from mitsuba.core import Transform as tf

from coll_detection import *

class Head:
    def __init__(self, icp): #icp is initial cranicm pointi
        self._cranicm_radius = 3.0
        self._neck_radius = 1.5
        self._neck_length = 6.0
        self._init_cranicm_point = icp
        self._init_neck_toWorld = tf.translate(Vector(icp.x, icp.y-3, icp.z)) * tf.scale(Vector(1, self._neck_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5))

        self.cranicm_prop = {
                'type' : 'sphere', 
                'center' : self._init_cranicm_point,
                'radius' : self._cranicm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

        self.neck_prop = {
                'type' : 'cylinder',
                'toWorld': self._init_neck_toWorld,
                'radius' : self._neck_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

    def horiz_move(self, gx, gy, gz):
        movement = tf.translate(Vector(gx, gy, gz)) # Affine matrix for horizontal movement.
        self._init_cranicm_point = movement * self._init_cranicm_point
        self._init_neck_toWorld = movement * self._init_neck_toWorld
        self.cranicm_prop['center'] = movement * self.cranicm_prop['center']
        self.neck_prop['toWorld'] = movement * self.neck_prop['toWorld']

    def calc_cranicm_point(self):
        return self.cranicm_prop['center']
    
    def get_cranicm_radius(self):
        return self._cranicm_radius

    def get_neck_radius(self):
        return self._neck_radius

    def get_neck_length(self):
        return self._neck_length

    def get_property_list(self):
        property_list = []
        property_list.append(self.cranicm_prop)
        property_list.append(self.neck_prop)

        return property_list

    def constr_json_data(self):
        # Construct JSON data
        json_data = {
            'true_position': {
                'head': dict(zip(['x', 'y', 'z'], self.calc_cranicm_point())) 
            }
        }
        return json_data

    def is_collision_to_capsule(self, p1, p2, radius):
        print 'Now, calculating the collision of head and capsule'
        min_dist = calc_min_dist_segment_point(p1, p2, self.calc_cranicm_point())
        return True if (min_dist <= radius + self.get_cranicm_radius()) else False

    def is_collision_to_sphere(self, p1, radius):
        print 'Now, calculating the collision of head and sphere'
        min_dist = calc_min_dist_point_point(p1, self.calc_cranicm_point())
        return True if (min_dist <= radius + self.get_cranicm_radius()) else False
