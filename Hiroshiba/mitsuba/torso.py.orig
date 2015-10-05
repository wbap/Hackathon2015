from mitsuba.core import *
from mitsuba.core import Transform as tf

from coll_detection import *

class Torso:
    def __init__(self, itp): # itp is initial torso point
        self._torso_radius = 5.5
        self._torso_length = 18.0
        self._init_clavile_toWorld = tf.translate(Vector(itp.x, itp.y+(self._torso_length/2), itp.z)) * tf.scale(Vector(self._torso_radius, 1, self._torso_radius)) * tf.rotate(Vector(1, 0, 0), -90)
        self._init_torso_cylinder_toWorld = tf.translate(Vector(itp.x, itp.y, itp.z)) * tf.scale(Vector(1, self._torso_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5)) 
        self._init_hip_toWorld = tf.translate(Vector(itp.x, itp.y-(self._torso_length/2), itp.z)) * tf.scale(Vector(self._torso_radius, 1, self._torso_radius)) * tf.rotate(Vector(1, 0, 0), 90)

        self._torso_cylinder_point = itp

        self.clavile_prop = {
                'type' : 'disk',
                'toWorld': self._init_clavile_toWorld,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }
        self.torso_cylinder_prop = {
                'type' : 'cylinder',
                'toWorld': self._init_torso_cylinder_toWorld,
                'radius' : self._torso_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                } 
            }
        self.hip_prop = {
                'type' : 'disk',
                'toWorld': self._init_hip_toWorld,
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
        self._init_clavile_toWorld = movement * self._init_clavile_toWorld
        self._init_torso_cylinder_toWorld = movement * self._init_torso_cylinder_toWorld
        self._init_hip_toWorld = movement * self._init_hip_toWorld

        self.clavile_prop['toWorld'] = movement * self.clavile_prop['toWorld']
        self.torso_cylinder_prop['toWorld'] = movement * self.torso_cylinder_prop['toWorld']
        self.hip_prop['toWorld'] = movement * self.hip_prop['toWorld']

        self._torso_cylinder_point = movement * self._torso_cylinder_point

    def calc_torso_point(self):
        return self._torso_cylinder_point

    def calc_torso_upper_point(self):
        return Point(self._torso_cylinder_point.x, self._torso_cylinder_point.y + self._torso_length/2, self._torso_cylinder_point.z)

    def calc_torso_lower_point(self):
        return Point(self._torso_cylinder_point.x, self._torso_cylinder_point.y - self._torso_length/2, self._torso_cylinder_point.z)

    def get_torso_radius(self):
        return self._torso_radius

    def get_torso_length(self):
        return self._torso_length

    def get_property_list(self):
        property_list = []
        property_list.append(self.clavile_prop)
        property_list.append(self.torso_cylinder_prop)
        property_list.append(self.hip_prop)

        return property_list

    def constr_json_data(self):
        # Construct JSON data
        json_data = {
            'true_position': {
                'torso': dict(zip(['x', 'y', 'z'], self.calc_torso_point())) 
            }
        }
        return json_data

    def is_collision_to_capsule(self, p1, p2, radius):
        print 'Now, calculating the collision of torso and capsule'
        min_dist = calc_min_dist_segment_segment(self.calc_torso_upper_point(), self.calc_torso_lower_point(), p1, p2)
        return True if (min_dist <= radius + self.get_torso_radius()) else False

    def is_collision_to_sphere(self, p1, radius):
        print 'Now, calculating the collision of torso and sphere'
        min_dist = calc_min_dist_segment_point(self.calc_torso_upper_point(), self.calc_torso_lower_point(), p1)
        return True if (min_dist <= radius + self.get_torso_radius()) else False
